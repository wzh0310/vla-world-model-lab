#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""兼容入口：规范实现已迁移至 vla/inference/infer_openvla_official_demo.py。"""
import os
import runpy

_canon = os.path.join(os.path.dirname(__file__), "..", "vla", "inference", "infer_openvla_official_demo.py")
runpy.run_path(_canon, run_name="__main__")
