# ---- Imports used throughout the notebook ----------------------------------------
import os, sys, json, time, glob, math, random, platform, warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=UserWarning)
%matplotlib inline
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "font.size": 9})

# ---- Fix every random seed we control --------------------------------------------
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
try:
    import torch
    torch.manual_seed(RANDOM_SEED)
    torch.use_deterministic_algorithms(False)   # CPU inference only; determinism not enforced in cuDNN
except Exception:
    torch = None


def _version(module_name: str) -> str:
    """Return an installed package version, or a short 'not installed' marker."""
    try:
        import importlib.metadata as md
        return md.version(module_name)
    except Exception:
        try:
            return __import__(module_name).__version__
        except Exception:
            return "not installed"


def _cpu_name() -> str:
    """Best-effort CPU model string, portable across OSes."""
    name = platform.processor() or ""
    if sys.platform == "win32":
        name = os.environ.get("PROCESSOR_IDENTIFIER", name)
    elif sys.platform.startswith("linux"):
        try:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
        except Exception:
            pass
    return name or "unknown"


env_rows = [
    ("Python",            sys.version.split()[0]),
    ("Platform",          f"{platform.system()} {platform.release()} ({platform.machine()})"),
    ("CPU",               _cpu_name()),
    ("Logical CPU cores", str(os.cpu_count())),
    ("Execution device",  "CPU (no GPU used anywhere in this notebook)"),
    ("Run timestamp",     time.strftime("%Y-%m-%d %H:%M:%S")),
    ("Random seed",       str(RANDOM_SEED)),
]
for pkg in ["numpy", "pandas", "opencv-python", "matplotlib", "torch",
            "torchvision", "ultralytics", "mediapipe", "scipy", "nbformat"]:
    env_rows.append((pkg, _version(pkg)))

env_df = pd.DataFrame(env_rows, columns=["Item", "Value"])
print("=" * 74)
print("HARDWARE / SOFTWARE CONFIGURATION  (recorded for reproducibility)")
print("=" * 74)
for item, value in env_rows:
    print(f"  {item:<20}: {value}")
env_df.to_csv(RESULTS_DIR / "environment.csv", index=False)
print("=" * 74)
print(f"Saved -> {RESULTS_DIR / 'environment.csv'}")
