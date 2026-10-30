---
title: PF2e Spell Tagger
emoji: 🔮
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# PF2e Spell Tagger

Write the **name, rank and description** of a homebrew Pathfinder 2e spell. The model fills in traits, traditions, save, area, damage types and sustained. Edit anything, then download a Foundry VTT JSON (*Import Data* in Foundry). Foundry JSON files can also be imported here, edited and exported again.

Models (trained on rank + description only): iteration 2 (NB-LR) and iteration 3 (NB-LR + `bge-small` transformer), from [kadzioriron/PF2e_spells_definders](https://huggingface.co/kadzioriron/PF2e_spells_definders).

## Files

| File | What it does |
|---|---|
| `app.py` | Streamlit front: input, editor, Foundry JSON import/export |
| `predictor.py` | loads the models from the model repo and predicts |
| `deploy.py` | uploads the models and this folder to Hugging Face |
| `Dockerfile`, `requirements.txt` | how the Space is built |

## Deploy (3 steps)

1. **Token:** <https://huggingface.co/settings/tokens> → *Create new token* → type **Write**. Keep it out of the code.
2. **Upload** (from the project folder, with the notebooks of iterations 2 and 3 already run):
   ```bash
   pip install huggingface_hub
   HF_TOKEN=hf_xxx python huggingface/deploy.py
   ```
   This puts `model.joblib`, `network.pt`, `ensemble.joblib` in the model repo and this folder in the Space. The Space then builds by itself (~5 min).
3. **Only if the model repo is private:** Space → *Settings* → *Variables and secrets* → *New secret* `HF_TOKEN` = a **Read** token.

Optional variable: `MODEL_REPO` (another model repo).

## Run locally

```bash
pip install -r huggingface/requirements.txt
cd huggingface && streamlit run app.py
```

Without internet access to the model repo, put the 3 model files in `huggingface/models/`.
