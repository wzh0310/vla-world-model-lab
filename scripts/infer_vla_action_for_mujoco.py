#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""兼容入口：规范实现已迁移至 vla/inference/infer_vla_action_for_mujoco.py。"""
import os
import runpy

_canon = os.path.join(os.path.dirname(__file__), "..", "vla", "inference", "infer_vla_action_for_mujoco.py")
runpy.run_path(_canon, run_name="__main__")
