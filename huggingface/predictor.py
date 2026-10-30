"""Iteration 2 (NB-LR) and iteration 3 (NB-LR + transformer ensemble), trained on rank + description only."""
import html
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from scipy import sparse
from sentence_transformers import SentenceTransformer
from torch import nn

MODEL_REPO = os.environ.get("MODEL_REPO", "kadzioriron/PF2e_spells_definders")
HF_TOKEN = os.environ.get("HF_TOKEN")  # only needed when the model repo is private
LOCAL_MODELS = Path(__file__).parent / "models"  # optional: put the files here to run without the Hub
VARIANT = "only_rank_and_description"
ONE_CHOICE = {"save", "area"}
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_CHUNK_WORDS = 200

# ---------------------------------------------------------------- text cleaning (same as the notebooks)
INLINE_PATTERN = re.compile(r"@(UUID|Damage|Check|Template|Embed|Localize)\[")
HTML_PATTERN = re.compile(r"<[^>]+>")
RANDOM_ID_PATTERN = re.compile(r"(?=.*\d)[A-Za-z0-9]{16}")


def closing_bracket(value: str, start: int) -> int:
    depth = 0
    for index in range(start, len(value)):
        if value[index] == "[":
            depth += 1
        elif value[index] == "]":
            depth -= 1
            if depth == 0:
                return index
    return len(value) - 1


def render_inline(kind: str, body: str, label: str | None) -> str:
    if label:
        return label
    if kind == "UUID":
        name = body.rsplit(".", 1)[-1]
        return " " if RANDOM_ID_PATTERN.fullmatch(name) else name
    if kind == "Damage":
        return " ".join(re.sub(r"[\[\],]", " ", body).split())
    if kind == "Check":
        parts = body.split("|")
        is_basic = any(part.split(":")[0] == "basic" for part in parts[1:])
        return f"basic {parts[0]} save" if is_basic else f"{parts[0]} check"
    if kind == "Template":
        params = dict(part.split(":", 1) for part in body.split("|") if ":" in part)
        shape = params.get("type", body.split("|")[0])
        return f"{params['distance']}-foot {shape}" if "distance" in params else shape
    return " "


def expand_inline(value: str) -> str:
    parts, position = [], 0
    while (match := INLINE_PATTERN.search(value, position)):
        body_end = closing_bracket(value, match.end() - 1)
        body = value[match.end():body_end]
        end, label = body_end, None
        if body_end + 1 < len(value) and value[body_end + 1] == "{":
            label_end = value.find("}", body_end + 1)
            end = len(value) - 1 if label_end == -1 else label_end
            label = value[body_end + 2:end]
        parts.append(value[position:match.start()])
        parts.append(f" {render_inline(match.group(1), body, label)} ")
        position = end + 1
    parts.append(value[position:])
    return "".join(parts)


def clean_text(value: str) -> str:
    value = HTML_PATTERN.sub(" ", expand_inline(value or ""))
    return " ".join(html.unescape(value).lower().split())


# ---------------------------------------------------------------- models
class MultiTaskNetwork(nn.Module):
    # Same layers as in iteration_3.ipynb, so the saved weights can be loaded.
    def __init__(self, input_dim: int, groups: dict[str, list[str]], hidden_dim: int = 256) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3))
        self.heads = nn.ModuleDict({group: nn.Linear(hidden_dim, len(labels)) for group, labels in groups.items()})

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared(features)
        return {group: head(shared) for group, head in self.heads.items()}


def model_file(name: str) -> str:
    local = LOCAL_MODELS / name
    return str(local) if local.exists() else hf_hub_download(MODEL_REPO, name, token=HF_TOKEN)


class SpellPredictor:
    def __init__(self) -> None:
        iteration2 = joblib.load(model_file("model.joblib"))
        self.vectorizer, self.groups, self.nb_lr = iteration2["vectorizer"], iteration2["groups"], iteration2[VARIANT]
        self.ensemble = joblib.load(model_file("ensemble.joblib"))[VARIANT]
        state = torch.load(model_file("network.pt"), map_location="cpu", weights_only=True)[VARIANT]
        self.network = MultiTaskNetwork(state["shared.0.weight"].shape[1], self.groups)
        self.network.load_state_dict(state)
        self.network.eval()
        self.sentence_encoder = SentenceTransformer(EMBEDDING_MODEL)

    def _nb_lr(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        encoder = self.nb_lr["encoder"]
        extra = encoder.transform(frame[encoder.feature_names_in_].astype(str))
        X = sparse.hstack([self.vectorizer.transform(frame["text"]), extra]).tocsr().astype(np.float64)
        return {group: np.column_stack([model.predict_proba(X.multiply(ratio).tocsr())[:, 1] for ratio, model in models])
                for group, models in self.nb_lr["models"].items()}

    def _network(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        words = frame["text"].iloc[0].split()
        pieces = [" ".join(words[start:start + EMBEDDING_CHUNK_WORDS]) for start in range(0, max(len(words), 1), EMBEDDING_CHUNK_WORDS)]
        vector = self.sentence_encoder.encode(pieces, normalize_embeddings=True).sum(axis=0, keepdims=True)
        vector /= np.linalg.norm(vector, axis=1, keepdims=True)
        encoder = self.ensemble["network_encoder"]
        extra = encoder.transform(frame[encoder.feature_names_in_].astype(str)).toarray()
        with torch.no_grad():
            outputs = self.network(torch.tensor(np.hstack([vector, extra]), dtype=torch.float32))
        return {group: (logits.softmax(dim=1) if group in ONE_CHOICE else logits.sigmoid()).numpy() for group, logits in outputs.items()}

    def predict(self, rank: int, description: str, iteration: int = 2) -> dict[str, list[str]]:
        """Only rank + description go in; every Foundry field the models know comes out."""
        frame = pd.DataFrame([{"text": clean_text(description), "rank": rank}])
        probabilities = self._nb_lr(frame)
        thresholds = self.nb_lr["thresholds"]
        if iteration == 3:
            network, weights = self._network(frame), self.ensemble["weights"]
            probabilities = {group: weights[group] * p + (1 - weights[group]) * network[group] for group, p in probabilities.items()}
            thresholds = self.ensemble["thresholds"]
        result = {}
        for group, p in probabilities.items():
            chosen = [int(p[0].argmax())] if group in ONE_CHOICE else np.flatnonzero(p[0] >= thresholds[group])
            result[group] = [self.groups[group][index] for index in chosen]
        return result
