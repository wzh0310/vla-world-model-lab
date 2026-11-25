# OpenVLA 环境依赖与模型准备说明

本文记录在本项目 `vla` conda 环境中准备 OpenVLA 推理环境的完整过程、版本锁定原因、模型权重来源与最简加载验证方法。所有操作在 13 服务器的 `wzh-ai` 容器内执行，模型权重不入库。

## 1. 环境分工与版本锁定

OpenVLA 的自定义模型代码来自 HuggingFace 仓库内的 `modeling_prismatic.py` / `processing_prismatic.py`，官方 `requirements-min.txt` 对三个核心包有**硬锁定**：

```
timm==0.9.10
tokenizers==0.19.1
transformers==4.40.1
torch>=2.2.0
```

因此不能简单地把 `transformers` 升到最新版；更高版本会破坏 `trust_remote_code` 加载路径。容器 `vla` 环境最终锁定的版本如下：

| 包 | 版本 | 说明 |
|---|---|---|
| Python | 3.12.16 | 容器既有环境 |
| torch | 2.6.0+cu124 | 容器既有，满足要求 |
| transformers | 4.40.1 | OpenVLA 硬锁定 |
| tokenizers | 0.19.1 | OpenVLA 硬锁定 |
| timm | 0.9.10 | OpenVLA 硬锁定 |
| peft | 0.11.1 | 官方 LoRA 脚本默认版本 |
| accelerate | 0.30.1 | 从 1.16.0 下调：新版 `dispatch_model` 会对 4bit 量化模型调用 `.to()` 而报错 |
| datasets | 3.6.0 | 与 huggingface-hub 0.36 兼容 |
| pyarrow | 26.0.1 | 已安装 |
| huggingface-hub | 0.36.0 | 满足 transformers 4.40.1 的 `<1.0` 约束 |
| pillow | 13.0.0 | 已升级 |
| opencv-python | 5.1.0.95 | 已升级 |
| diffusers | 0.30.3 | 从 0.43.0 下调，消除 hub 版本冲突（项目未使用该库） |

安装命令（容器 `vla` 环境）：

```bash
PIP=/opt/miniconda3/envs/vla/bin/pip
$PIP install --upgrade accelerate peft pillow opencv-python datasets pyarrow
$PIP install transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10 \
    peft==0.11.1 datasets==3.6.0 huggingface_hub==0.36.0 \
    accelerate==0.30.1 diffusers==0.30.3
/opt/miniconda3/envs/vla/bin/pip check   # No broken requirements found
```

## 2. 模型权重准备

目标模型：`openvla/openvla-7b`（旗舰版，DINOv2+SigLIP 双视觉编码器 + Llama-2 7B）。

| 项目 | 值 |
|---|---|
| 本地路径 | `/data/wzh/robot/models/openvla-7b` |
| 仓库修订 | `47a0ec7fc4ec123775a391911046cf33cf9ed83f` |
| 文件总数 | 18 |
| 总体积 | 15.09 GB |
| 权重分片 | 6.95 GB + 6.97 GB + 1.16 GB（safetensors） |
| 是否入库 | 否，权重存放于仓库外的 `/data/wzh/robot/models/` |

### 下载方式选择

- HuggingFace 官方源可达，但容器到官方 CDN 的大文件单连接只有约 0.03–0.23 MB/s，15 GB 需要十几小时；Xet 组件已安装但同样受限。
- `hf-mirror.com` 在该容器内 DNS 解析不稳定，且 resolve 请求返回 403。
- 最终使用 ModelScope 镜像 `zixiaoBios/openvla-7b`（文件列表和字节数与官方一致），三分片并行下载，实测聚合约 7–9 MB/s，约 25 分钟完成。

```bash
/opt/miniconda3/envs/vla/bin/pip install modelscope
/opt/miniconda3/envs/vla/bin/modelscope download \
    --model zixiaoBios/openvla-7b \
    --local_dir /data/wzh/robot/models/openvla-7b
```

下载完成后应核对三个 safetensors 分片大小分别为 6948961960 / 6971232040 / 1162406824 字节。

## 3. 最简加载验证

仓库脚本 `scripts/verify_openvla_load.py` 负责验证处理器与模型初始化：

```bash
VLA=/opt/miniconda3/envs/vla/bin/python
$VLA scripts/verify_openvla_load.py \
    --model-dir /data/wzh/robot/models/openvla-7b \
    --predict
```

该脚本会：打印环境与版本 → 检查关键文件 → 初始化 `AutoProcessor` → 以 bf16 加载 `AutoModelForVision2Seq`（`trust_remote_code=True`）→ 打印参数量、dtype 与显存占用 → 构造 224×224 随机图像和指令 prompt，执行一次 `predict_action` 并打印 7 维动作。

低显存卡可追加 `--quant int4`，使用 nf4 + 双重量化加载。

### 实测验证结果（A6000 48GB）

| 模式 | 加载耗时 | 参数量 | dtype | 显存分配/保留 | 单步推理 | 动作 shape |
|---|---|---|---|---|---|---|
| bf16 | 11.7 s | 7.54B | bfloat16 | 15.13 / 15.17 GB | 0.96 s | (7,) |
| int4 nf4 | 35.0 s | 3.90B（量化后统计口径） | float16 | 4.38 / 4.55 GB | 1.31 s | (7,) |

两种模式均成功返回 7 维动作，处理器初始化约 1.5 s。int4 路径的两个关键细节已固化到脚本：

1. 量化模型必须显式传 `device_map`，且加载后不能再调用 `.to()`；
2. int4 下视觉编码器权重为 fp16，输入需同为 fp16，否则会触发 `Input type (BFloat16) and bias type (Half)` 错误。

## 4. 已知约束

- OpenVLA 依赖源于 Llama-2，权重继承 Llama Community License；代码为 MIT。
- 模型只支持单张 224×224 图像，prompt 模板严格为 `In: What action should the robot take to {指令}?\nOut:`，动作是 7 维末端增量，`predict_action` 返回值已按 `unnorm_key` 反归一化。
- `vla` 环境现在锁定了旧版 transformers 生态；若其他项目需要新版 transformers，应另建 conda 环境，不要直接升级本环境。
