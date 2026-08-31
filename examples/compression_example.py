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
# # 特徴量圧縮タスクのサンプル(diabetes データセット)
#
# scikit-learn の `load_diabetes`(10項目の生理学的指標から糖尿病の1年後の
# 進行度を予測する回帰問題)を題材に、あらかじめこの回帰タスクで学習して
# おいた PyTorch モデルの中間層(ボトルネック層)の出力を教師信号として、
# 同じ振る舞いを目指して特徴量圧縮用の SAMACT モデルを学習し、推論・評価・
# 保存/復元までの一連の流れを示します。元モデルを回帰タスクで学習済みに
# しておくことで、ボトルネック層の出力が意味のある特徴量表現になった状態を
# 作ります。
#
# さらに、SAMACT が再現したボトルネック出力を新しい回帰モデルの入力として
# 再学習し、元のボトルネック出力で学習した場合と比べてどの程度の精度に
# なるかも確認します。

# %%
import torch
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from samact_model_adapter.api import compress_from_pytorch, load_from_artifact
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
# ## 2. 元になる PyTorch モデルを回帰タスクで学習する
#
# 最終層の手前にある `Linear(16, 4)` の出力(4次元)をボトルネック層として
# 使います(`bottleneck_layer_name` を省略すると、出力層候補を除いた
# `nn.Linear` のうち `out_features` が最小の層として自動検出されます)。

# %%
torch.manual_seed(0)

model = nn.Sequential(
    nn.Linear(INPUT_DIM, 16),
    nn.ReLU(),
    nn.Linear(16, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
loss_fn = nn.MSELoss()

model.train()
for _ in range(500):
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
config = SAMACTConfig(tC=32, hidden_units=8, epochs=20, torch_seed=0)

# %% [markdown]
# ## 4. ボトルネック層の出力を教師信号として SAMACT モデルを学習する
#
# `bottleneck_layer_name` を省略すると自動検出されます。名前を明示したい場合は
# `bottleneck_layer_name="2"` のように指定します。特徴量圧縮モードでは元モデルの
# `y` は使わないため、`train_loader` に含まれる回帰の目的変数はここでは
# 無視されます。

# %%
adapter = compress_from_pytorch(model, train_loader, config=config)

print(f"mode={adapter.mode}, task={adapter.task}")
print(
    f"bottleneck_layer_name={adapter.bottleneck_layer_name}, "
    f"bottleneck_dim={adapter.bottleneck_dim}"
)
print(f"mse_per_epoch={adapter.fit_result.mse_per_epoch}")

# %% [markdown]
# ## 5. 推論
#
# 戻り値は shape `(N, bottleneck_dim)` の `np.ndarray` です。

# %%
sample = x_test[0:1].numpy()
compressed = adapter.predict(sample)
print(f"compressed feature: {compressed}")

# %% [markdown]
# ## 6. 評価データでの比較評価
#
# 元モデルを学習済みにしてあるので、そのボトルネック出力を SAMACT モデルが
# どれだけ再現できているかを比較できます。

# %%
metrics = adapter.evaluate(model, test_loader)
print(metrics)

# %% [markdown]
# ## 7. SAMACT が再現したボトルネック出力で回帰モデルを再学習する
#
# 元の PyTorch モデルのボトルネック出力(正解)で学習した回帰モデルと、SAMACT
# が再現したボトルネック出力で学習した回帰モデルとで、評価データに対する
# MSE を比較します。両方とも同じ単層の線形回帰モデル(`nn.Linear(bottleneck_dim,
# 1)`)を新規に学習し、ボトルネック層より後ろの元モデルの重みは再利用しません。

# %%
def _extract_original_bottleneck(inputs: torch.Tensor) -> torch.Tensor:
    bottleneck_layers = nn.Sequential(*list(model.children())[:3])  # Linear(16, 4) まで
    with torch.no_grad():
        return bottleneck_layers(inputs)


def _train_regressor_head(features: torch.Tensor, targets: torch.Tensor) -> nn.Module:
    torch.manual_seed(0)
    head = nn.Linear(adapter.bottleneck_dim, 1)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.02)
    loss_fn = nn.MSELoss()
    head.train()
    for _ in range(500):
        optimizer.zero_grad()
        loss = loss_fn(head(features), targets)
        loss.backward()
        optimizer.step()
    head.eval()
    return head


def _regressor_mse(head: nn.Module, features: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        return nn.functional.mse_loss(head(features), targets).item()


original_train_features = _extract_original_bottleneck(x_train)
original_test_features = _extract_original_bottleneck(x_test)
original_head = _train_regressor_head(original_train_features, y_train)
original_head_mse = _regressor_mse(original_head, original_test_features, y_test)

samact_train_features = torch.tensor(adapter.predict(x_train.numpy()), dtype=torch.float32)
samact_test_features = torch.tensor(adapter.predict(x_test.numpy()), dtype=torch.float32)
samact_head = _train_regressor_head(samact_train_features, y_train)
samact_head_mse = _regressor_mse(samact_head, samact_test_features, y_test)

print(f"original bottleneck -> regressor test MSE: {original_head_mse:.2f}")
print(f"SAMACT bottleneck   -> regressor test MSE: {samact_head_mse:.2f}")

# %% [markdown]
# ## 8. artifact として保存・復元

# %%
ARTIFACT_DIR = "artifact_compression"

adapter.save_artifact(ARTIFACT_DIR, overwrite=True)
restored = load_from_artifact(ARTIFACT_DIR)

print(restored.predict(sample))
