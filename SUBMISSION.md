# ANYmal C Navigation Submission

本仓库是基于 MotrixLab 的 ANYmal C 导航训练代码包，包含可直接运行的学生环境 `student-anymal-c-navigation`、环境配置、XML 模型资产、PPO 训练配置和运行脚本。

## 运行环境

- Python 3.10
- UV
- MotrixLab workspace packages: `motrix_envs`, `motrix_rl`
- 推荐使用支持 CUDA 的 PyTorch 环境进行训练

安装依赖：

```bash
uv sync --all-packages --all-extras
```

如果只使用 PyTorch + SKRL：

```bash
uv sync --all-packages --extra skrl-torch
```

## 关键代码位置

- 环境注册与配置：`motrix_envs/src/motrix_envs/navigation/anymal_c/cfg.py`
- 环境闭环逻辑：`motrix_envs/src/motrix_envs/navigation/anymal_c/anymal_c_np.py`
- XML 场景和机器人资产：`motrix_envs/src/motrix_envs/navigation/anymal_c/xmls/`
- RL 训练参数：`motrix_rl/src/motrix_rl/tasks/student_anymal_c.py`
- 环境包导入入口：`motrix_envs/src/motrix_envs/navigation/__init__.py`

## 环境检查

查看环境：

```bash
uv run scripts/view.py --env student-anymal-c-navigation --sim-backend np --num-envs 1
```

检查 CUDA：

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.version.cuda)"
```

## 训练

小规模验证：

```bash
uv run scripts/train.py --env student-anymal-c-navigation --sim-backend np --train-backend torch --rllib skrl --num-envs 64
```

正式训练示例：

```bash
uv run scripts/train.py --env student-anymal-c-navigation --sim-backend np --train-backend torch --rllib skrl --num-envs 1024
```

训练日志默认保存到：

```text
runs/student-anymal-c-navigation/
```

## TensorBoard

训练过程中或训练完成后均可打开：

```bash
uv run tensorboard --logdir runs/student-anymal-c-navigation
```

浏览器访问：

```text
http://localhost:6006
```

## 推理

使用训练好的最优权重进行推理：

```bash
uv run scripts/play.py --env student-anymal-c-navigation --sim-backend np --rllib skrl --num-envs 1 --policy runs/student-anymal-c-navigation/skrl/<RUN_NAME>/checkpoints/best_agent.pt
```

如果没有 `best_agent.pt`，可选择 `checkpoints/` 下步数最大的 `agent_*.pt`。

## 提交说明

提交代码包时应包含完整 MotrixLab 工程代码和本学生环境相关文件。不要提交本地虚拟环境、训练日志、缓存和临时文件：

- 不提交：`.venv/`
- 不提交：`runs/`
- 不提交：`__pycache__/`
- 不提交：`.codex`
- 不提交：本地 TensorBoard 截图之外的大型 checkpoint，除非课程明确要求

如果需要提交训练结果截图或报告，建议单独放在 `report/` 或课程平台指定位置，不要混入源码目录。
