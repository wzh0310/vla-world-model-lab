# -*- coding: utf-8 -*-
"""快速筛选能成功的同义指令（默认场景，不渲染，仅闭环推理）。"""
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path("/data/wzh/robot/vla-world-model-lab")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"vla"/"utils"))
import importlib.util
_s = importlib.util.spec_from_file_location("va", ROOT/"vla/utils/vla_adapter.py")
_va = importlib.util.module_from_spec(_s); _s.loader.exec_module(_va)

MUJOCO_SITE = "/opt/miniconda3/envs/mujoco/lib/python3.12/site-packages"
if os.path.isdir(MUJOCO_SITE) and MUJOCO_SITE not in sys.path:
    sys.path.append(MUJOCO_SITE)
sys.path.insert(0, "/data/wzh/robot/mujoco-studio-lab")
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from envs.push_cube_env import PushCubeEnv

MODEL_DIR = "/data/wzh/robot/models/openvla-7b"
ADAPTER = ROOT/"results/vla_finetune_tuned/best_adapter"
CANDIDATES = [
    "push the cube into the red marked area",
    "push the cube into the red target zone",
    "put the cube on the red marker",
    "move the cube to the red target",
]
processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
model = AutoModelForVision2Seq.from_pretrained(MODEL_DIR, quantization_config=bnb,
    device_map={"":0}, low_cpu_mem_usage=True, trust_remote_code=True)
meta = json.load(open(ADAPTER/"finetune_meta.json"))
lo = [float(x) for x in meta["norm_stats"]["min"]]; hi = [float(x) for x in meta["norm_stats"]["max"]]
key = meta["unnorm_key"]
model.config.norm_stats[key] = {"action": {
    "q01": lo[:3]+[0,0,0]+[lo[3]], "q99": hi[:3]+[0,0,0]+[hi[3]],
    "mask":[True,True,True,False,False,False,True]}}
model = PeftModel.from_pretrained(model, str(ADAPTER)); model.eval()

def predict(rgb, instruction):
    pil,_ = _va.preprocess_image(rgb)
    inp = processor(_va.build_prompt(instruction), pil).to("cuda:0", dtype=torch.float16)
    with torch.inference_mode():
        raw = model.predict_action(**inp, unnorm_key=key, do_sample=False)
        torch.cuda.synchronize()
    r = np.asarray(raw,dtype=np.float64).reshape(-1)[:7]
    return [float(r[0]),float(r[1]),float(r[2]),float((r[6]+1.0)/2.0)]

scenes = json.load(open(ROOT/"results/vla_generalization/scenes.json"))["scenes"]
scene = next(s for s in scenes if s["kind"]=="instruction" and s["scene_id"]==0)
for instr in CANDIDATES:
    env = PushCubeEnv(scene_override=True, obs_mode="both", rgb_preset="demo",
        render_camera="global_cam", render_mode="rgb_array", max_episode_steps=50)
    obs, info = env.reset(options={"scene": scene})
    d0 = float(info["cube_target_dist"]); succ=False
    for t in range(50):
        a = predict(obs["image"], instr)
        obs, reward, term, trunc, info = env.step(np.asarray(a))
        # 与录制脚本一致：每帧调用 render（可能影响时序/状态），保证筛选条件相同
        env.render()
        succ = bool(info.get("success", False))
        if term or trunc: break
    de = float(info["cube_target_dist"])
    print(f"SCREEN succ={int(succ)} dist {d0:.4f}->{de:.4f} instr={instr!r}", flush=True)
    env.close()
