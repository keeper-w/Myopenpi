# 本地复现 Franka + LIBERO + pi0.5

本文档说明如何在本机使用 RTX A6000 运行以下完整链路：

```text
LIBERO / MuJoCo Franka Panda 仿真
    -> 相机图像、机器人状态、语言指令
    -> WebSocket 客户端
    -> pi0.5-LIBERO 策略服务器
    -> 7 维末端执行器动作
    -> Franka 仿真环境
```

项目根目录假设为：

```text
/home/dell/wzm/openpi
```

策略服务器使用已有的 `openpi_env`，LIBERO 仿真客户端使用独立的
`openpi_libero` Python 3.8 环境。

## 1. 需要下载什么

| 内容 | 用途 | 是否必须 | 大小 |
|---|---|---:|---:|
| `pi05_libero` checkpoint | pi0.5 仿真推理 | 是 | 约 11.58 GiB |
| LIBERO 源码和仿真资源 | Franka Panda、任务和初始状态 | 是 | 较小 |
| `physical-intelligence/libero` | 重新训练或微调 | 否 | 约 32.54 GiB |
| `openvla/modified_libero_rlds` | 从原始 RLDS 重新转换数据 | 否 | 约 9.53 GiB |

只运行官方 checkpoint 的 Franka 仿真不需要下载训练数据集。

pi0.5 的 10,000+ 小时预训练数据没有随本仓库完整公开。公开可下载的是
LIBERO 微调数据集和训练完成的 `pi05_libero` checkpoint。

建议仅推理时预留至少 30 GiB；如果还要下载训练数据和保存训练 checkpoint，
建议预留 100 至 150 GiB。

## 2. 检查策略服务器环境

激活已有环境：

```bash
conda activate openpi_env
cd /home/dell/wzm/openpi
```

检查 Python、CUDA、JAX 和 LeRobot：

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "import jax; print(jax.__version__); print(jax.devices())"
python -c "import lerobot.common.datasets.lerobot_dataset; print('LeRobot OK')"
```

预期能看到 RTX A6000、一个 CUDA 设备以及 `LeRobot OK`。

如果 `lerobot.common` 不存在，安装本仓库锁定版本：

```bash
python -m pip uninstall -y lerobot

GIT_LFS_SKIP_SMUDGE=1 python -m pip install --no-deps \
  "lerobot @ git+https://github.com/huggingface/lerobot.git@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
```

手动下载链接：

- https://github.com/huggingface/lerobot/archive/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5.zip

下载并解压后，也可以在源码目录执行：

```bash
python -m pip install --no-deps .
```

## 3. 下载 pi0.5-LIBERO 官方权重

官方 checkpoint 地址：

```text
gs://openpi-assets/checkpoints/pi05_libero
```

官方对象列表：

- https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints%2Fpi05_libero%2F

### 3.1 使用 OpenPI 下载器

这是推荐方式，会把权重放到项目内的 `.openpi_cache`：

```bash
conda activate openpi_env
cd /home/dell/wzm/openpi

export OPENPI_DATA_HOME=/home/dell/wzm/openpi/.openpi_cache
unset OPENPI_LIBERO_CHECKPOINT

python -c 'from openpi.shared.download import maybe_download; print(maybe_download("gs://openpi-assets/checkpoints/pi05_libero"))'
```

下载约 11.58 GiB。完成后的目录应为：

```text
/home/dell/wzm/openpi/.openpi_cache/openpi-assets/checkpoints/pi05_libero
```

### 3.2 使用 Google Cloud CLI

也可以使用 `gcloud`：

```bash
cd /home/dell/wzm/openpi
mkdir -p .openpi_cache/openpi-assets/checkpoints

gcloud storage cp --recursive \
  gs://openpi-assets/checkpoints/pi05_libero \
  .openpi_cache/openpi-assets/checkpoints/
```

或者使用 `gsutil`：

```bash
gsutil -m cp -r \
  gs://openpi-assets/checkpoints/pi05_libero \
  /home/dell/wzm/openpi/.openpi_cache/openpi-assets/checkpoints/
```

### 3.3 验证权重

```bash
cd /home/dell/wzm/openpi

test -d .openpi_cache/openpi-assets/checkpoints/pi05_libero/params \
  && echo "params OK"

test -f .openpi_cache/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json \
  && echo "norm stats OK"

du -sh .openpi_cache/openpi-assets/checkpoints/pi05_libero
```

必须同时看到 `params OK` 和 `norm stats OK`。

如果出现 `Checkpoint directory does not exist`，说明设置了
`OPENPI_LIBERO_CHECKPOINT`，但下载尚未完成或路径不正确。下载前应先执行：

```bash
unset OPENPI_LIBERO_CHECKPOINT
```

## 4. 下载 LIBERO Franka 仿真环境

仓库通过 Git 子模块固定 LIBERO 版本。推荐执行：

```bash
cd /home/dell/wzm/openpi
git submodule update --init third_party/libero
```

如果 GitHub 访问较慢，可以手动下载固定版本：

- https://github.com/Lifelong-Robot-Learning/LIBERO/archive/f78abd68ee283de9f9be3c8f7e2a9ad60246e95c.zip

将压缩包内容解压到：

```text
/home/dell/wzm/openpi/third_party/libero
```

验证关键仿真资源：

```bash
cd /home/dell/wzm/openpi

test -d third_party/libero/libero/libero/bddl_files && echo "BDDL OK"
test -d third_party/libero/libero/libero/init_files && echo "init states OK"
test -d third_party/libero/libero/libero/assets && echo "assets OK"
```

三个检查都应显示 `OK`。

## 5. 创建 LIBERO 客户端环境

LIBERO 的依赖版本较老，不应直接安装到 `openpi_env`。创建独立 Python 3.8 环境：

```bash
conda create -n openpi_libero python=3.8 -y
conda activate openpi_libero

cd /home/dell/wzm/openpi

python -m pip install \
  -r examples/libero/requirements.txt \
  -r third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113

python -m pip install -e packages/openpi-client
python -m pip install -e third_party/libero
```

验证客户端依赖：

```bash
python -c "import libero; print('LIBERO OK')"
python -c "import robosuite; print('robosuite OK')"
python -c "import mujoco; print('MuJoCo OK')"
python -c "import openpi_client; print('OpenPI client OK')"
```

客户端只负责 Franka 仿真、图像渲染和发送观测，VLA 推理由 `openpi_env`
中的策略服务器完成。

## 6. 启动 pi0.5 策略服务器

打开终端 1：

```bash
conda activate openpi_env
cd /home/dell/wzm/openpi

export OPENPI_DATA_HOME=/home/dell/wzm/openpi/.openpi_cache
export OPENPI_LIBERO_CHECKPOINT=/home/dell/wzm/openpi/.openpi_cache/openpi-assets/checkpoints/pi05_libero

bash examples/libero/run_local_policy_server.sh
```

策略服务器将检查 checkpoint、在 RTX A6000 上加载 pi0.5-LIBERO，并在
`0.0.0.0:8000` 启动 WebSocket 服务。

另开一个终端检查：

```bash
nvidia-smi
ss -lntp | grep 8000
```

等待模型完全加载并开始监听端口后，再启动 Franka 客户端。

## 7. 启动 Franka Panda 仿真

打开终端 2：

```bash
conda activate openpi_libero
cd /home/dell/wzm/openpi

export LIBERO_TASK_SUITE=libero_spatial
export LIBERO_NUM_TRIALS=1
export OPENPI_POLICY_HOST=127.0.0.1
export OPENPI_POLICY_PORT=8000
export MUJOCO_GL=egl

bash examples/libero/run_local_client.sh
```

脚本默认每个任务只评测一次，适合第一次冒烟测试。回放视频保存在：

```text
/home/dell/wzm/openpi/data/libero/videos/
```

LIBERO 可选的离线 demonstrations 目录默认为：

```text
/home/dell/wzm/openpi/data/libero/datasets/
```

运行脚本会自动创建该目录；策略评测不需要在其中放置训练数据。

如果 EGL 初始化失败，切换到 GLX：

```bash
export MUJOCO_GL=glx
bash examples/libero/run_local_client.sh
```

GLX 模式通常需要有效的 `DISPLAY` 和 X11 会话。

## 8. 切换 LIBERO 任务集

支持以下五个任务集：

```text
libero_spatial
libero_object
libero_goal
libero_10
libero_90
```

例如运行 LIBERO-10：

```bash
export LIBERO_TASK_SUITE=libero_10
export LIBERO_NUM_TRIALS=1
bash examples/libero/run_local_client.sh
```

确认整个流程稳定后，将每个任务的评测次数设置为 50：

```bash
export LIBERO_NUM_TRIALS=50
bash examples/libero/run_local_client.sh
```

仓库报告的 pi0.5-LIBERO 结果为：

| 任务集 | 成功率 |
|---|---:|
| LIBERO-Spatial | 98.8% |
| LIBERO-Object | 98.2% |
| LIBERO-Goal | 98.0% |
| LIBERO-10 | 92.4% |
| 平均 | 96.85% |

## 9. 下载官方 LIBERO 训练数据集（可选）

只运行官方 checkpoint 的 Franka 仿真可以跳过本节。

OpenPI 的 `pi05_libero` 配置使用转换好的 LeRobot 数据集：

- https://huggingface.co/datasets/physical-intelligence/libero

该数据集约 32.54 GiB，包含 LIBERO-Spatial、LIBERO-Object、
LIBERO-Goal 和 LIBERO-10，当前 LeRobot 环境应使用 `v2.0` 修订。

下载到 LeRobot 能直接识别的位置：

```bash
conda activate openpi_env
cd /home/dell/wzm/openpi

python -m pip install -U "huggingface_hub[cli]"

export HF_LEROBOT_HOME=/home/dell/wzm/openpi/data/lerobot

hf download physical-intelligence/libero \
  --repo-type dataset \
  --revision v2.0 \
  --local-dir /home/dell/wzm/openpi/data/lerobot/physical-intelligence/libero
```

验证数据集：

```bash
test -f data/lerobot/physical-intelligence/libero/meta/info.json \
  && echo "metadata OK"

du -sh data/lerobot/physical-intelligence/libero
```

以后运行数据加载、归一化统计或训练时，都必须设置相同的缓存目录：

```bash
export HF_LEROBOT_HOME=/home/dell/wzm/openpi/data/lerobot
```

## 10. 原始 RLDS 数据（通常不需要）

原始修改版 LIBERO RLDS 数据位于：

- https://huggingface.co/datasets/openvla/modified_libero_rlds

该数据集约 9.53 GiB。只有需要重新执行
`examples/libero/convert_libero_data_to_lerobot.py` 时才需要下载。OpenPI 已提供
转换完成的 `physical-intelligence/libero`，普通训练和评测不需要重复转换。

## 11. A6000 训练注意事项

RTX A6000 的 48 GiB 显存足以运行 pi0.5 推理。仓库当前的 `pi05_libero`
配置是全参数训练配置，默认全局 batch size 为 256，单张 A6000 很可能显存不足。

建议先完成官方 checkpoint 的仿真评测。如果需要重新训练，应单独增加 pi0.5
低显存或 LoRA 配置，并降低 batch size；不要直接在单张 A6000 上启动默认
`pi05_libero` 全参数训练。

## 12. 常用排查命令

检查 GPU：

```bash
nvidia-smi
```

检查策略端口：

```bash
ss -lntp | grep 8000
```

检查 checkpoint：

```bash
find .openpi_cache/openpi-assets/checkpoints/pi05_libero -maxdepth 4 -type f | head
```

检查 LIBERO 子模块：

```bash
git submodule status third_party/libero
```

检查生成的视频：

```bash
find data/libero/videos -maxdepth 1 -type f -name '*.mp4'
```
