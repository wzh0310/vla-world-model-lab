# configs

本目录集中管理世界模型实验室的实验超参数。

## default.yaml（推荐，单一权威来源）

`configs/default.yaml` 是当前唯一推荐的配置文件，按 **data / model / training / eval** 四段组织，
覆盖数据口径、RSSM/AE 模型维度、AE 与世界模型训练超参、评估集设置。所有数值均为已验证配置。

训练脚本用 `--config` 读取，**命令行显式参数优先**：

```bash
# 世界模型训练：全部超参来自 default.yaml
python trainer/train_world_model.py --config configs/default.yaml

# 在配置基础上只覆盖 max-steps（其余仍来自配置）
python trainer/train_world_model.py --config configs/default.yaml --max-steps 5000
```

`@eval.windows_per_traj` / `@eval.gif_steps` 等评估参数由 `trainer/eval_single_step.py` 与
`trainer/make_prediction_gif.py` 的对应命令行项使用（默认值与配置一致）。

### 快速修改高频参数

以下三个参数改动频率最高，直接从 `default.yaml` 对应字段修改即可，训练/评估脚本自动读取：

| 配置字段 | 含义 | 约束 | 影响 |
|---|---|---|---|
| `data.image_size` | 图像分辨率 | 16 的倍数（32/64/96…） | 改变数据口径，需重训 AE 与 WM |
| `data.seq_len` | 训练序列长度 | 正整数 | 改变时间窗口，影响 GRU 记忆跨度 |
| `model.rssm.stoch_dim` | 随机隐状态维度 | 正整数 | 改变特征维度（det+stoch），解码器自动适配 |

> 分辨率可配置依赖 `models/vae.py` 的自适应池化 + 输出 resize（向后兼容改造）：
> `ConvEncoder` 把卷积特征固定到 4×4，64 输入时为恒等；`ConvDecoder` 默认 `out_size=64`，
> 输出后按需 resize。因此在 64 分辨率下历史 checkpoint 权重可原样加载；
> 改成 32/96 等分辨率后必须从零重训 AE，再用新 AE 训练 WM。

## world_model.yaml（早期配置，保留兼容）

`configs/world_model.yaml` 为 AE 训练阶段使用的早期配置，`trainer/train_ae.py --config` 仍兼容。
新实验请统一以 `default.yaml` 为准。

## 归一化口径（两份配置共用）

- 图像：`[0,255] -> [-1,1]`，224×224 双线性下采样到 64×64
- 动作：按动作空间 `[low, high]` 线性映射到 `[-1,1]`
- 奖励：`r / 10` 后裁剪到 `[-1,1]`
