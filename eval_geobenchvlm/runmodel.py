#!/usr/bin/env python3
"""run_models.py

Usage
-----
python runmodel.py <model-key> [any other args the target script expects]

Example:
    python runmodel.py llava1pt5 --data_path /datasets/GEOBench-VLM

Supported model keys to script files
----------------------------------
llava1pt5 = llava1pt5_cls_single.py
llava1pt6 = llava1pt6_cls_single.py
llavaone1 = llavaone1_cls_single.py
qwen      = qwen_cls_single.py
internvl  = internvl_cls_single.py
"""

import subprocess
import sys
from pathlib import Path

MODEL2SCRIPT = {
    "llava1pt5": "llava1pt5_cls_single.py",
    "llava1pt6": "llava1pt6_cls_single.py",
    "llavaone1": "llavaone1_cls_single.py",
    "qwen": "qwen_cls_single.py",
    "internvl": "internvl_cls_single.py",
}

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_models.py <model-key> [extra args...]\n",
              "Model‑keys:", ", ".join(MODEL2SCRIPT), file=sys.stderr)
        sys.exit(1)

    model_key, *extra = sys.argv[1:]
    script_name = MODEL2SCRIPT.get(model_key)
    if script_name is None:
        print(f"✗ Unknown model-key '{model_key}'. Choose one of:",
              ", ".join(MODEL2SCRIPT), file=sys.stderr)
        sys.exit(1)

    # Make sure the script exists next to this dispatcher.
    if not Path(script_name).exists():
        print(f"✗ Script '{script_name}' not found in current directory.", file=sys.stderr)
        sys.exit(1)

    # Forward execution to the underlying script (inherits stdout/stderr).
    subprocess.run([sys.executable, script_name, *extra], check=True)

if __name__ == "__main__":
    main()
