#!/usr/bin/env python3
"""run_temporal.py

Usage:
    python run_temporal.py <model-key> [extra args...]

Example:
    python run_temporal.py qwen /datasets/GEOBench-VLM

Model keys:
    qwen        → qwen_cls_temporal.py
    llavaone1   → llavaone1_cls_temporal.py
"""

import subprocess
import sys
from pathlib import Path

MODEL2SCRIPT = {
    "qwen": "qwen_cls_temporal.py",
    "llavaone1": "llavaone1_cls_temporal.py",
}

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_temporal.py <model-key> [extra args...]\n",
              "Model-keys:", ", ".join(MODEL2SCRIPT), file=sys.stderr)
        sys.exit(1)

    model_key, *extra = sys.argv[1:]
    script_name = MODEL2SCRIPT.get(model_key)
    if script_name is None:
        print(f"✗ Unknown model-key '{model_key}'. Choose one of:",
              ", ".join(MODEL2SCRIPT), file=sys.stderr)
        sys.exit(1)

    if not Path(script_name).exists():
        print(f"✗ Script '{script_name}' not found in current directory.", file=sys.stderr)
        sys.exit(1)

    subprocess.run([sys.executable, script_name, *extra], check=True)

if __name__ == "__main__":
    main()