# SAMACT Model Adapter
# Copyright (C) 2026 Maviss Design
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
#
# If you modify this software and make it available over a network,
# you must provide the corresponding source code to users (AGPLv3 §13).
#
# For commercial licensing options, contact:
# solution-sales@maviss-design.com
#
# See the LICENSE file for full details.

# %% [markdown]
# # 回帰タスクのサンプル(diabetes データセット)
#
# scikit-learn の `load_diabetes`(10 項目の生理学的指標から糖尿病の1年後の
# 進行度を予測する回帰問題)を題材に、あらかじめ学習しておいた PyTorch 回帰
# モデルを元に、同じ振る舞いを目指して SAMACT モデルを standalone モードで
# 学習し、推論・評価・保存/復元までの一連の流れを示します。元モデルを
# 学習済みにしておくことで、SAMACT モデルが元モデルの予測をどれだけ
# 再現できているかを比較できます。

# %%
import torch
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from samact_model_adapter.api import load_from_artifact, translate_from_pytorch
from samact_model_adapter.config import SAMACTConfig

# %% [markdown]
# ## 1. diabetes データセットを用意する
#
# `load_diabetes` の入力は既に平均0・分散1程度に標準化済みです。
# 学習用 80% / 評価用 20% に分割します。

# %%
diabetes = load_diabetes()
INPUT_DIM = diabetes.data.shape[1]

x = torch.tensor(diabetes.data, dtype=torch.float32)
y = torch.tensor(diabetes.target, dtype=torch.float32).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=32)

# %% [markdown]
# ## 2. 元になる PyTorch モデルを学習する
#
# SAMACT 側の学習とは別に、通常の PyTorch の手順(optimizer + loss.backward())
# で先に学習しておきます。

# %%
torch.manual_seed(0)

model = nn.Sequential(
    nn.Linear(INPUT_DIM, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

model.train()
for _ in range(300):
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    train_mse = loss_fn(model(x_train), y_train).item()
print(f"pytorch model train MSE: {train_mse:.2f}")

# %% [markdown]
# ## 3. SAMACT 側の学習設定
#
# 省略した項目にはデフォルト値が使われます。

# %%
config = SAMACTConfig(tC=32, hidden_units=16, epochs=20, torch_seed=0)

# %% [markdown]
# ## 4. 元モデルと同じ振る舞いを目指して SAMACT モデルを学習する

# %%
adapter = translate_from_pytorch(model, train_loader, config=config, task="regression")

print(f"mode={adapter.mode}, task={adapter.task}, output_dim={adapter.output_dim}")
print(f"mse_per_epoch={adapter.fit_result.mse_per_epoch}")

# %% [markdown]
# ## 5. 推論
#
# 戻り値は shape `(N, output_dim)` の `np.ndarray` です。

# %%
sample = x_test[0:1].numpy()
predicted = adapter.predict(sample)
print(f"predicted value: {predicted}, true value: {y_test[0].item()}")

# %% [markdown]
# ## 6. 評価データでの比較評価
#
# 元モデル(PyTorch)と SAMACT モデルの、評価データに対する MSE / MAE を
# それぞれ確認します。

# %%
metrics = adapter.evaluate(model, test_loader)
print(metrics)

# %% [markdown]
# ## 7. artifact として保存・復元

# %%
ARTIFACT_DIR = "artifact_regression"

adapter.save_artifact(ARTIFACT_DIR, overwrite=True)
restored = load_from_artifact(ARTIFACT_DIR)

print(restored.predict(sample))
