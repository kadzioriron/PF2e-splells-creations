"""Uploads the models to the model repo and this folder to the Space.

Run: HF_TOKEN=hf_xxx python huggingface/deploy.py   (or log in once with `hf auth login` and leave HF_TOKEN out)
"""
import os
from pathlib import Path

from huggingface_hub import HfApi

MODEL_REPO = "kadzioriron/PF2e_spells_definders"
SPACE_REPO = "kadzioriron/PF2E"
HERE = Path(__file__).parent
ARTIFACTS = HERE.parent / "iterations"
MODEL_FILES = {
    "model.joblib": ARTIFACTS / "iteration_2" / "artifacts" / "model.joblib",       # iteration 2 (NB-LR)
    "network.pt": ARTIFACTS / "iteration_3" / "artifacts" / "network.pt",           # iteration 3 network
    "ensemble.joblib": ARTIFACTS / "iteration_3" / "artifacts" / "ensemble.joblib",  # iteration 3 weights + thresholds
}

api = HfApi(token=os.environ.get("HF_TOKEN"))
for name, path in MODEL_FILES.items():
    print(f"uploading {name} -> {MODEL_REPO}")
    api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=MODEL_REPO, repo_type="model")
print(f"uploading the app -> {SPACE_REPO}")
api.upload_folder(folder_path=HERE, repo_id=SPACE_REPO, repo_type="space",
                  ignore_patterns=["models/*", "__pycache__/*", "deploy.py", ".gitignore"])
print(f"done: https://huggingface.co/spaces/{SPACE_REPO}")
