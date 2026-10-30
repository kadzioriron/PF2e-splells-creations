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


labels = predictor.groups
with gr.Blocks(title="PF2e Spell Tagger") as demo:
    base = gr.State(None)  # the imported Foundry file, so its other fields are kept on export
    gr.Markdown("# 🔮 PF2e Spell Tagger\nWrite a spell → the model fills in the Foundry fields → fix anything → "
                "download the JSON and import it in Foundry VTT (*Import Data*).")

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### 1. Your spell")
            with gr.Row():
                name = gr.Textbox(EMPTY["name"], label="Name", scale=3)
                rank = gr.Slider(1, 10, value=1, step=1, label="Rank", scale=2)
            description = gr.Textbox(label="Description", lines=6,
                                     placeholder="A roaring blast of fire detonates at a spot you designate, dealing 6d6 fire damage...")
            with gr.Row():
                model = gr.Radio(list(MODELS), value=list(MODELS)[0], label="Model (reads only rank + description)", scale=3)
                fill_button = gr.Button("✨ Fill in the fields", variant="primary", scale=1)
        with gr.Column(scale=1):
            gr.Markdown("### Import")
            upload = gr.File(label="Foundry spell (.json)", file_types=[".json"], type="filepath")

    gr.Markdown("### 2. Check and edit")
    with gr.Row():
        with gr.Column():
            traits = gr.Dropdown(labels["traits"], multiselect=True, allow_custom_value=True, label="Traits")
            traditions = gr.Dropdown(labels["traditions"], multiselect=True, label="Traditions")
            rarity = gr.Dropdown(RARITIES, value="common", label="Rarity")
        with gr.Column():
            save = gr.Dropdown(SAVES, value="none", label="Saving throw")
            basic = gr.Checkbox(label="Basic save")
            area = gr.Dropdown(AREAS, value="none", label="Area")
            area_value = gr.Number(0, label="Area size (feet)", precision=0)
            sustained = gr.Checkbox(label="Sustained")
        with gr.Column():
            time = gr.Textbox("2", label="Cast time (1, 2, 3, reaction, free, 1 minute...)")
            range_ = gr.Textbox(label="Range")
            duration = gr.Textbox(label="Duration")
            target = gr.Textbox(label="Targets")
            folder = gr.Dropdown(list(FOLDERS), value="Rank 1", label="Foundry folder")
    damage = gr.Dataframe(headers=["type", "formula"], datatype=["str", "str"], row_count=(1, "dynamic"), column_count=(2, "fixed"),
                          interactive=True, label="Damage (type, formula like 6d6)")

    gr.Markdown("### 3. Download")
    export_button = gr.Button("⬇️ Make Foundry JSON", variant="primary")
    with gr.Row():
        file = gr.File(label="Foundry JSON")
        preview = gr.Code(language="json", label="JSON")

    editor = [name, rank, description, traits, traditions, rarity, save, basic, area, area_value,
              sustained, time, range_, duration, target, folder, damage]
    fill_button.click(fill_in, [model, *editor], editor)
    upload.upload(import_json, upload, [base, *editor])
    export_button.click(export_json, [base, *editor], [file, preview])
    gr.Examples([["Lava Geyser", 4, "You channel magical energy to rupture the earth, summoning a burst of molten rock. "
                  "The eruption deals 4d6 fire damage to anyone in a 20-foot burst. Each creature in the area must attempt a basic Reflex save."],
                 ["Dread Gaze", 2, "You make a target creature believe it is being watched by a terrifying presence. "
                  "The target must attempt a Will save. On a failure it becomes frightened 2."]],
                [name, rank, description])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="orange"))
