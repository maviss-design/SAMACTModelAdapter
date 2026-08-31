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
# # 分類タスクのサンプル(digits データセット)
#
# scikit-learn の `load_digits`(8x8 手書き数字、10 クラス)を題材に、
# あらかじめ学習しておいた PyTorch 分類モデルを元に、同じ振る舞いを目指して
# SAMACT モデルを standalone モードで学習し、推論・評価・保存/復元までの
# 一連の流れを示します。ハイパーパラメータ(`eta`, `iota`, `decay_period`,
# `epochs` など)は、PySAMACT の `Fit()` を直接呼び出す既存のサンプル
# シナリオ(`SampleScenarioDigit.ipynb`)に合わせています。

# %%
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from samact_model_adapter.api import load_from_artifact, translate_from_pytorch
from samact_model_adapter.config import LearningPropertyConfig, SAMACTConfig

# %% [markdown]
# ## 1. digits データセットを用意する
#
# 各画素値(0-16)を `[0, 1]` へ正規化し、学習用 80% / 評価用 20% に分割します。

# %%
digits = load_digits()
NUM_CLASSES = 10
INPUT_DIM = digits.data.shape[1]

x = torch.tensor(digits.data, dtype=torch.float32) / 16.0
y = torch.tensor(digits.target, dtype=torch.long)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

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
    nn.Linear(INPUT_DIM, 32),
    nn.ReLU(),
    nn.Linear(32, NUM_CLASSES),
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

model.train()
for _ in range(100):
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    train_accuracy = (model(x_train).argmax(dim=1) == y_train).float().mean().item()
print(f"pytorch model train accuracy: {train_accuracy:.3f}")

# %% [markdown]
# ## 3. SAMACT 側の学習設定
#
# `eta=5`, `iota=2`, `decay_period=1`, `epochs=10` は参照元シナリオと同じ値です。

# %%
config = SAMACTConfig(
    tC=32,
    hidden_units=32,
    epochs=10,
    torch_seed=0,
    learning_property=LearningPropertyConfig(eta=5, iota=2, decay_period=1),
)

# %% [markdown]
# ## 4. 元モデルと同じ振る舞いを目指して SAMACT モデルを学習する

# %%
adapter = translate_from_pytorch(model, train_loader, config=config, task="classification")

print(f"mode={adapter.mode}, task={adapter.task}")
print(f"accuracy_per_epoch={adapter.fit_result.accuracy_per_epoch}")

# %% [markdown]
# ## 5. 推論
#
# 1件のみ渡すと int、複数件を渡すと `np.ndarray` が返ります。

# %%
sample = x_test[0:1].numpy()
predicted_class = adapter.predict(sample)
print(f"predicted class: {predicted_class}, true class: {int(y_test[0])}")

# %% [markdown]
# ## 6. 評価データでの比較評価
#
# 元モデル(PyTorch)と SAMACT モデルの、評価データ正解率をそれぞれ確認します。

# %%
metrics = adapter.evaluate(model, test_loader)
print(metrics)

# %% [markdown]
# ## 7. artifact として保存・復元

# %%
ARTIFACT_DIR = "artifact_classification_digits"

adapter.save_artifact(ARTIFACT_DIR, overwrite=True)
restored = load_from_artifact(ARTIFACT_DIR)

print(restored.predict(sample))
