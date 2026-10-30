# PF2e Spell Metadata Tagger

You write the description of a homebrew Pathfinder 2e spell, and the model fills in the fields Foundry VTT needs: **traits, traditions, saving throw, area, damage types and sustained**.

## Results

All iterations are scored on the same 192 test spells. **Score** = micro-F1 of traits + traditions (1.0 = perfect). Always guessing `concentrate, manipulate` scores 0.473.

| Iteration | Model | Input | Score | |
|---|---|---|---|---|
| 0 | TF-IDF + kNN | description | 0.38 * | ❌ |
| 1 | BiGRU neural network | rank + description | 0.388 | ❌ |
| 1.5 | BiGRU + fixes + Naive Bayes | rank + description + category, cantrip, casting time | 0.740 | ⚠️ |
| 2 | Naive Bayes + logistic regression (NB-LR) | rank + description + category, cantrip, casting time | **0.786** | ✅ final |
| 3 🤖 | transformer + ensemble with iteration 2 | rank + description + category, cantrip, casting time | 0.787 | ⚠️ no gain |

\* old dataset, not comparable. 🤖 Iteration 3 was made entirely with AI.

## Test: the original idea – only rank + description

The original idea: the user types **only the rank and the description**, and the model fills in the rest. Iterations 1-3 were tested this way on the same 192 test spells. Iteration 0 is kept as the original code (old dataset), so it is not in this test.

| | 1 BiGRU | 1.5 BiGRU + NB | 2 NB-LR | 3 ensemble 🤖 |
|---|---|---|---|---|
| **Score** | 0.388 | 0.647 | **0.720** | 0.719 |
| traits | – | 0.734 ¹ | 0.777 | 0.777 |
| traditions | – | 0.620 | 0.661 | 0.667 |
| save (accuracy) | – | 0.849 | 0.891 | 0.875 |
| area (accuracy) | – | 0.833 | 0.833 | 0.823 |
| damage types | – | 0.280 | 0.724 | 0.724 |
| sustained | – | 0.431 | 0.556 | 0.556 |

– = this iteration does not predict the field. ¹ without `cantrip` and `focus`: in 1.5 they come from a rule that needs the category, so without it they are not predicted (they still count as misses in the score).

**What the test shows:**

* **The idea works.** With only rank + description, iteration 2 scores 0.720 (with all inputs 0.786).
* What is **written in the description** stays almost the same: save, area, damage types, traits like fire, mental or healing.
* The loss comes from what is **not in the text**: whether it is a **focus spell** or a **cantrip**, and so the traditions (focus spells have none).
* One extra question to the user, *"spell, cantrip or focus spell?"*, would bring back most of the difference.

Notebooks 1-3 end with this test: a score table and the predictions for the same 5 example spells.

## Iterations

| | Model | Libraries | How text becomes numbers |
|---|---|---|---|
| **0** | kNN: copies the labels of the 3 most similar spells | scikit-learn | TF-IDF word weights |
| **1** | BiGRU network, trained from scratch | PyTorch | word vectors learned from scratch |
| **1.5** | BiGRU of iteration 1 + Naive Bayes hints | PyTorch, numpy | word vectors + Naive Bayes word scores |
| **2** | Naive Bayes + logistic regression (NB-LR) | scikit-learn | bag of words: 0/1 for every word and word pair |
| **3** 🤖 | transformer `bge-small` + small network, combined with iteration 2 | sentence-transformers, PyTorch | pretrained transformer: 384 numbers per text |

Notebooks 1-3 start with a short table that explains that iteration. Iteration 0 is the original code.

### What each iteration taught us

* **0 – kNN:** a quick proof of concept on an older dataset. Copying labels from similar spells is not enough.
* **1 – BiGRU:** worse than a simple guess. The targets were wrong (keywords copied from the text), the data was too small to learn word meanings from scratch, and the cleaning threw away useful words.
* **1.5 – fixed BiGRU:** the same network with the fixes jumps from 0.39 to 0.740. The fixes mattered more than the model.
* **2 – NB-LR:** the simplest model gives the best result. Naive Bayes finds the key words (`disbelieve` → illusion, `you teleport` → teleportation), and logistic regression learns how much to trust them. One small model per label, the same code for all 6 fields.
* **3 – transformer (made with AI):** the same score as iteration 2 for much more code and a 130 MB model. Iteration 2 stays the final model.

## Folders

```
data/spellsdata/         Foundry spell export (1,994 JSON files)
eda/eda.ipynb            data analysis
iterations/
  iteration_0/           iteration_0.ipynb (+ old dataset and saved model)
  iteration_1/           iteration_1.ipynb + artifacts/
  iteration_1_5/         iteration_1_5.ipynb + artifacts/
  iteration_2/           iteration_2.ipynb + artifacts/   (final model)
  iteration_3/           iteration_3.ipynb + artifacts/   (made with AI)
documents/               proposal, design document, diagram
huggingface/             Gradio app for the Hugging Face Space (iterations 2 and 3), see huggingface/README.md
```

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Open each notebook from its own folder. Every notebook runs on its own, in any order. The first run of `iteration_3` downloads the transformer (~130 MB).

## Limitations

* The test set is small (192 spells), so differences of ±0.01 are noise.
* Class traits like `witch` or `cleric` are not in the description and must be chosen by the user.
* The damage labels in Foundry are incomplete, which limits the damage-type score.
