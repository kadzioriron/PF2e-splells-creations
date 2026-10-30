"""Gradio front: name + rank + description in, editable Foundry VTT spell JSON out."""
import spaces  # must be the first import on ZeroGPU

import json
import re
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from foundry import AREAS, EMPTY, FOLDERS, RARITIES, SAVES, build_spell, from_prediction, read_spell
from predictor import SpellPredictor


@spaces.GPU(duration=1)
def _noop() -> None:
    """ZeroGPU needs one GPU function. The models are small and run on the CPU, so it is never called."""


predictor = SpellPredictor()
MODELS = {"Iteration 2 – NB-LR (fast, main model)": 2, "Iteration 3 – NB-LR + transformer": 3}
# Order of the editable fields, the same as the components below.
FIELDS = ["name", "rank", "description", "traits", "traditions", "rarity", "save", "basic", "area", "area_value",
          "sustained", "time", "range", "duration", "target", "folder", "damage"]


def to_components(fields: dict) -> list:
    values = [fields[key] for key in FIELDS[:-1]]
    return values + [pd.DataFrame(fields["damage"], columns=["type", "formula"])]


def to_fields(values: tuple) -> dict:
    fields = dict(zip(FIELDS, values))
    damage = fields["damage"] if isinstance(fields["damage"], pd.DataFrame) else pd.DataFrame(columns=["type", "formula"])
    fields["damage"] = [row for row in damage.fillna("").astype(str).to_dict("records") if row.get("type")]
    fields["traits"], fields["traditions"] = fields["traits"] or [], fields["traditions"] or []
    return fields


def fill_in(model: str, *values) -> list:
    """Predicts traits, traditions, save, area, damage types and sustained from the rank and the description."""
    fields = to_fields(values)
    if not fields["description"].strip():
        raise gr.Error("Write a description first.")
    prediction = predictor.predict(int(fields["rank"]), fields["description"], MODELS[model])
    return to_components(from_prediction(fields, prediction))


def import_json(path: str | None) -> list:
    """Loads a Foundry spell JSON into the editor."""
    if not path:
        return [gr.update()] * (len(FIELDS) + 1)
    spell = json.loads(Path(path).read_text(encoding="utf-8"))
    return [spell] + to_components(read_spell(spell))


def export_json(base: dict | None, *values) -> tuple[str, str]:
    """Builds the Foundry JSON file from the editor."""
    spell = build_spell(to_fields(values), base)
    name = re.sub(r"[^a-z0-9]+", "-", spell["name"].lower()).strip("-") or "spell"
    path = Path(tempfile.mkdtemp()) / f"{name}.json"
    text = json.dumps(spell, indent=4, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")
    return str(path), text


CSS = """
.gradio-container { max-width: 1000px !important; margin: auto; }
.hero { text-align: center; padding: 8px 0 4px; }
.hero h1 { font-size: 2.2em; margin-bottom: 4px; }
.hero p { opacity: .75; font-size: 1.05em; }
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 12px 0 4px; }
.steps div { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 12px 14px; font-size: .95em; }
.steps b { display: block; font-size: 1.05em; margin-bottom: 2px; color: var(--color-accent); }
.step-title h2 { margin: 18px 0 0; }
.hint { opacity: .7; font-size: .9em; }
@media (max-width: 640px) { .steps { grid-template-columns: 1fr; } }
"""

HERO = """
<div class="hero">
  <h1>🔮 PF2e Spell Tagger</h1>
  <p>Turn your homebrew Pathfinder 2e spell into a Foundry VTT item in a minute.</p>
</div>
<div class="steps">
  <div><b>1. Describe</b>Give the spell a name, a rank and the text a player would read.</div>
  <div><b>2. Check</b>The model fills in traits, save, area, damage… You can change everything.</div>
  <div><b>3. Download</b>Get a JSON file and use <i>Import Data</i> on any spell in Foundry.</div>
</div>
"""

ABOUT = """
**What the models do.** They read only the **rank** and the **description**, the way you would write a spell yourself.

* **Iteration 2 – NB-LR** (recommended): Naive Bayes finds the words that point to each trait (*disbelieve* → illusion), logistic regression decides how much to trust them. Fast and the most accurate.
* **Iteration 3 – NB-LR + transformer**: adds a small pretrained language model. A bit slower, about the same quality.

**What to double-check.** The description does not say whether a spell is a **cantrip** or a **focus spell**, so the model sometimes guesses `focus` for normal spells. Check the traits and the Foundry folder before downloading. Class traits like `wizard` or `cleric` are never predicted — add them yourself.

**Importing in Foundry.** Create (or right-click) a spell in the *Items* tab or in a compendium → *Import Data* → choose the downloaded file. The JSON uses the folder ids of the PF2e spell compendium export, so the spell lands in Cantrip / Rank N / Focus / Rituals.
"""

labels = predictor.groups
with gr.Blocks(title="PF2e Spell Tagger") as demo:
    base = gr.State(None)  # the imported Foundry file, so its other fields are kept on export
    gr.HTML(HERO)

    # ---------------------------------------------------------------- step 1
    gr.Markdown("## ✍️ 1. Describe your spell", elem_classes="step-title")
    with gr.Group():
        with gr.Row():
            name = gr.Textbox(EMPTY["name"], label="Spell name", scale=3)
            rank = gr.Slider(1, 10, value=1, step=1, label="Rank", scale=2)
        description = gr.Textbox(label="Description", lines=5, max_lines=14,
                                 placeholder="A roaring blast of fire detonates at a spot you designate, dealing 6d6 fire damage. "
                                             "Each creature in a 20-foot burst must attempt a basic Reflex save.")
    gr.Examples([["Lava Geyser", 4, "You channel magical energy to rupture the earth, summoning a burst of molten rock. "
                  "The eruption deals 4d6 fire damage to anyone in a 20-foot burst. Each creature in the area must attempt a basic Reflex save."],
                 ["Dread Gaze", 2, "You make a target creature believe it is being watched by a terrifying presence. "
                  "The target must attempt a Will save. On a failure it becomes frightened 2."]],
                [name, rank, description], label="No idea? Try an example")
    with gr.Row(equal_height=True):
        model = gr.Radio(list(MODELS), value=list(MODELS)[0], label="Model", scale=3,
                         info="Both read only the rank and the description.")
        fill_button = gr.Button("✨ Fill in the fields", variant="primary", size="lg", scale=1)
    status = gr.Markdown()
    with gr.Accordion("📂 Already have a Foundry spell file? Import it to edit", open=False):
        upload = gr.File(label="Foundry spell (.json)", file_types=[".json"], type="filepath", height=120)

    # ---------------------------------------------------------------- step 2
    gr.Markdown("## 🔍 2. Check and edit", elem_classes="step-title")
    gr.Markdown("Everything here can be changed. Type into *Traits* to add your own trait.", elem_classes="hint")
    with gr.Row():
        traits = gr.Dropdown(labels["traits"], multiselect=True, allow_custom_value=True, label="Traits", scale=2)
        traditions = gr.Dropdown(labels["traditions"], multiselect=True, label="Traditions", scale=1,
                                 info="Focus spells have none")
    with gr.Row():
        save = gr.Dropdown(SAVES, value="none", label="Saving throw")
        basic = gr.Checkbox(label="Basic save", info="damage halved on success")
        area = gr.Dropdown(AREAS, value="none", label="Area")
        area_value = gr.Number(0, label="Area size (feet)", precision=0)
    with gr.Row():
        damage = gr.Dataframe(headers=["type", "formula"], datatype=["str", "str"], row_count=(1, "dynamic"),
                              column_count=(2, "fixed"), interactive=True, label="Damage (type + dice, e.g. fire · 6d6)", scale=2)
        with gr.Column(scale=1):
            sustained = gr.Checkbox(label="Sustained", info="needs Sustain every turn")
            folder = gr.Dropdown(list(FOLDERS), value="Rank 1", label="Foundry folder")
    with gr.Accordion("More details: cast time, range, duration, targets, rarity", open=False):
        with gr.Row():
            time = gr.Textbox("2", label="Cast time", info="1, 2, 3, reaction, free, 1 minute…")
            range_ = gr.Textbox(label="Range", placeholder="30 feet")
            duration = gr.Textbox(label="Duration", placeholder="1 minute")
        with gr.Row():
            target = gr.Textbox(label="Targets", placeholder="1 creature")
            rarity = gr.Dropdown(RARITIES, value="common", label="Rarity")

    # ---------------------------------------------------------------- step 3
    gr.Markdown("## 📦 3. Download for Foundry", elem_classes="step-title")
    export_button = gr.Button("⬇️ Make Foundry JSON", variant="primary", size="lg")
    file = gr.File(label="Your file — click the file name to download", height=70)
    with gr.Accordion("Show the JSON", open=False):
        preview = gr.Code(language="json", label="JSON")

    with gr.Accordion("ℹ️ How it works and what to check", open=False):
        gr.Markdown(ABOUT)

    editor = [name, rank, description, traits, traditions, rarity, save, basic, area, area_value,
              sustained, time, range_, duration, target, folder, damage]
    fill_button.click(lambda: "⏳ Reading your spell…", None, status) \
        .then(fill_in, [model, *editor], editor) \
        .success(lambda: "✅ Done! Check the fields below — especially **traits** and the **Foundry folder**.", None, status)
    upload.upload(import_json, upload, [base, *editor]).success(lambda: "📂 File imported. Edit it below and download it again.", None, status)
    export_button.click(export_json, [base, *editor], [file, preview])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="orange", radius_size="lg"), css=CSS, show_error=True)
