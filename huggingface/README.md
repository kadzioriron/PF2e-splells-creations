---
title: PF2e Spell Tagger
emoji: 🔮
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 8501
suggested_hardware: cpu-basic
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
| `deploy.py` | uploads the models and this folder to Hugging Face, sets the Space to the free CPU |
| `Dockerfile`, `requirements.txt` | how the Space is built |
| `../.github/workflows/deploy-space.yml` | runs `deploy.py` on every push to GitHub |

## Deploy through GitHub (one-time setup)

1. **Hugging Face token:** <https://huggingface.co/settings/tokens> → *Create new token* → type **Write**.
2. **Put it in GitHub:** repo → *Settings* → *Secrets and variables* → *Actions* → *New repository secret*, name `HF_TOKEN`, value = the token.
3. **Push to `main`** (with `iterations/iteration_2/artifacts` and `iterations/iteration_3/artifacts` committed). The *Actions* tab shows the deploy; it can also be started there with *Run workflow*. The Space then builds by itself (~5 min).
4. **Only if the model repo is private:** Space → *Settings* → *Variables and secrets* → *New secret* `HF_TOKEN` = a **Read** token.

After that every push that changes `huggingface/` or the model files deploys again. By hand it also works: `HF_TOKEN=hf_xxx python huggingface/deploy.py`.

Optional variable: `MODEL_REPO` (another model repo).

## Run locally

```bash
pip install -r huggingface/requirements.txt
cd huggingface && streamlit run app.py
```

Without internet access to the model repo, put the 3 model files in `huggingface/models/`.
