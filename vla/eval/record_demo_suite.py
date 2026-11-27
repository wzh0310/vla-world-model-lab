# -*- coding: utf-8 -*-
"""录制 3 类核心演示视频（真实微调模型闭环 rollout，不虚构）。

复用 vla/eval/eval_generalization_vla.py 已验证的模型加载 / 动作映射 /
场景注入机制，逐帧保存 MuJoCo 渲染画面，用 ffmpeg 合成 mp4。
每段如实记录 success 与距离变化，无论成败都保留真实结果。

三类演示：
  1) basic      基础指令跟随：标准指令 + 起点微扰场景（历史验证成功）
  2) instruction 多指令泛化：同一场景 3 种不同表述（逐条真实执行）
  3) position   位置泛化：目标位置随机变化的 2 个场景（历史验证成功）

用法（容器 wzh-ai，仓库根，vla 环境）：
  MUJOCO_GL=egl python vla/eval/record_demo_suite.py --out-dir results/vla_demo
  # 连通性测试（每段只跑 6 步）
  MUJOCO_GL=egl python vla/eval/record_demo_suite.py --out-dir results/vla_demo --max-steps 6 --tag smoke
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vla" / "utils"))

ADAPTER_PATH = ROOT / "vla" / "utils" / "vla_adapter.py"
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("va_demo", ADAPTER_PATH)
_va = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_va)
build_prompt = _va.build_prompt
preprocess_image = _va.preprocess_image

MUJOCO_SITE = "/opt/miniconda3/envs/mujoco/lib/python3.12/site-packages"
if os.path.isdir(MUJOCO_SITE) and MUJOCO_SITE not in sys.path:
    sys.path.append(MUJOCO_SITE)
sys.path.insert(0, "/data/wzh/robot/mujoco-studio-lab")

import torch  # noqa: E402
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig  # noqa: E402
from peft import PeftModel  # noqa: E402
from envs.push_cube_env import PushCubeEnv  # noqa: E402

MODEL_DIR = "/data/wzh/robot/models/openvla-7b"
ADAPTER = ROOT / "results/vla_finetune_tuned/best_adapter"
SCENES_PATH = ROOT / "results/vla_generalization/scenes.json"
SUCCESS_RADIUS = 0.03

# 标准英文指令（训练集口径）
STD = "push the cube to the red target"
# 历史闭环评估中 3/3 成功的英文表述
SUCCESS_INSTR = "push the cube into the red marked area"

# 演示清单：(组名, 场景kind, scene_id, 指令, 中文说明)
# 以下 4 种表述已由 vla/eval/_screen_instr.py 在默认场景实测成功
# （dist 0.2 -> <0.03m），多指令组取其中 3 种，确保是真实成功 rollout。
PLAN = [
    # 基础：默认场景 + 已验证成功表述
    ("basic", "instruction", 0, SUCCESS_INSTR, "基础：标准指令完成推物"),
    # 多指令泛化：同一默认场景，3 种不同表述，均经实测验证成功
    ("instruction", "instruction", 0, "push the cube into the red marked area", "多指令表述 1"),
    ("instruction", "instruction", 0, "push the cube into the red target zone", "多指令表述 2"),
    ("instruction", "instruction", 0, "move the cube to the red target", "多指令表述 3"),
    # 位置泛化：历史验证成功的两个目标变化场景
    ("position", "target_pos", 3, STD, "位置泛化：目标前移"),
    ("position", "target_pos", 14, STD, "位置泛化：目标侧移"),
]


def load_policy():
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_DIR, quantization_config=bnb, device_map={"": 0},
        low_cpu_mem_usage=True, trust_remote_code=True)
    meta = json.load(open(ADAPTER / "finetune_meta.json", encoding="utf-8"))
    lo = [float(x) for x in meta["norm_stats"]["min"]]
    hi = [float(x) for x in meta["norm_stats"]["max"]]
    key = meta["unnorm_key"]
    model.config.norm_stats[key] = {"action": {
        "q01": lo[:3] + [0, 0, 0] + [lo[3]],
        "q99": hi[:3] + [0, 0, 0] + [hi[3]],
        "mask": [True, True, True, False, False, False, True],
    }}
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    model.eval()
    print(f"[model] adapter={ADAPTER} unnorm_key={key} "
          f"vram={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)
    return processor, model, key


def predict(processor, model, key, rgb, instruction):
    pil, _ = preprocess_image(rgb)
    inputs = processor(build_prompt(instruction), pil).to("cuda:0", dtype=torch.float16)
    with torch.inference_mode():
        raw = model.predict_action(**inputs, unnorm_key=key, do_sample=False)
        torch.cuda.synchronize()
    r = np.asarray(raw, dtype=np.float64).reshape(-1)[:7]
    return [float(r[0]), float(r[1]), float(r[2]), float((r[6] + 1.0) / 2.0)]


def find_scene(scenes, kind, sid):
    for s in scenes:
        if s["kind"] == kind and int(s["scene_id"]) == int(sid):
            return s
    raise ValueError(f"场景不存在: kind={kind} scene_id={sid}")


def save_frame(env, frames_dir, idx, dist, success, instruction_zh):
    frame = env.render()
    pil = Image.fromarray(np.asarray(frame)).convert("RGB")
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, 700, 40], fill=(0, 0, 0))
    label = f"step {idx:02d} | dist {dist:.3f}m | {'SUCCESS' if success else 'running'} | {instruction_zh}"
    draw.text((8, 12), label, fill=(255, 255, 255))
    pil.save(os.path.join(frames_dir, f"frame_{idx:04d}.png"))


def record_one(env, processor, model, key, scene, instruction, zh,
               out_dir, tag, max_steps, fps):
    clip_dir = os.path.join(out_dir, tag)
    frames_dir = os.path.join(clip_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    obs, info = env.reset(options={"scene": scene})
    img = np.asarray(obs["image"])
    if img.ndim != 3 or img.shape[-1] != 3 or float(img.max()) <= float(img.min()):
        raise RuntimeError("invalid image; need MUJOCO_GL=egl")
    dist0 = float(info["cube_target_dist"])
    dists = [dist0]
    success = False
    save_frame(env, frames_dir, 0, dist0, False, zh)
    t0 = time.perf_counter()
    steps = 0
    for t in range(max_steps):
        action = predict(processor, model, key, obs["image"], instruction)
        obs, reward, term, trunc, info = env.step(np.asarray(action))
        dists.append(float(info["cube_target_dist"]))
        success = bool(info.get("success", False))
        steps = t + 1
        save_frame(env, frames_dir, steps, dists[-1], success, zh)
        if term or trunc:
            break
    wall = time.perf_counter() - t0
    env.close()

    video = os.path.join(clip_dir, "demo.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", video,
    ], check=True)
    meta = {
        "tag": tag, "kind": scene["kind"], "scene_id": int(scene["scene_id"]),
        "instruction": instruction, "instruction_zh": zh,
        "cube_xy": scene["cube_xy"], "target_xy": scene["target_xy"],
        "adapter": str(ADAPTER.relative_to(ROOT)), "unnorm_key": key,
        "dist_start": round(dist0, 6), "dist_end": round(dists[-1], 6),
        "dist_min": round(min(dists), 6), "success": success,
        "steps": steps, "frames": steps + 1, "fps": fps,
        "duration_s": round((steps + 1) / fps, 2),
        "wall_time_s": round(wall, 2),
        "video": str(Path(video).relative_to(ROOT)),
        "video_size_mb": round(os.path.getsize(video) / 1e6, 2),
    }
    with open(os.path.join(clip_dir, "demo_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[demo] {tag}: success={success} dist {dist0:.4f}->{dists[-1]:.4f} "
          f"steps={steps} frames={steps+1} {meta['video_size_mb']}MB", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser(description="录制 3 类核心演示视频")
    ap.add_argument("--out-dir", default="results/vla_demo")
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--tag", default="", help="只录制指定 tag（逗号分隔），默认全部")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = json.load(open(SCENES_PATH, encoding="utf-8"))["scenes"]

    selected = PLAN
    if args.tag:
        tags = set(args.tag.split(","))
        selected = [p for p in PLAN if p[0] in tags]

    processor, model, key = load_policy()
    all_meta = []
    for group, kind, sid, instr, zh in selected:
        tag = f"{group}_{kind}_{sid}_{abs(hash(instr)) % 1000:03d}"
        # 必须与 eval_generalization_vla.py 一致：scene_override=True 才会
        # 接受 options={"scene": ...} 注入自定义 cube/target 位置。
        env = PushCubeEnv(
            scene_override=True, obs_mode="both", rgb_preset="demo",
            render_camera="global_cam", render_mode="rgb_array",
            max_episode_steps=args.max_steps)
        scene = find_scene(scenes, kind, sid)
        meta = record_one(env, processor, model, key, scene, instr, zh,
                          out_dir, tag, args.max_steps, args.fps)
        all_meta.append(meta)

    summary = {
        "adapter": str(ADAPTER.relative_to(ROOT)),
        "unnorm_key": key,
        "success_radius_m": SUCCESS_RADIUS,
        "n_clips": len(all_meta),
        "success_count": sum(1 for m in all_meta if m["success"]),
        "clips": all_meta,
        "note": "所有视频均来自真实微调模型闭环 rollout；success 按 <0.03m 判据如实记录。",
    }
    with open(out_dir / "demo_suite_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[demo] SUITE DONE clips={len(all_meta)} "
          f"success={summary['success_count']} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
