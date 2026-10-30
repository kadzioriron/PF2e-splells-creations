"""Streamlit front: name + rank + description in, editable Foundry VTT spell JSON out."""
import copy
import html
import json
import re
import secrets

import pandas as pd
import streamlit as st

from predictor import SpellPredictor, clean_text

st.set_page_config(page_title="PF2e Spell Tagger", page_icon="🔮", layout="wide")

# Folder ids from the Foundry export (data/spellsdata/_folders.json): the spell lands in the right folder on import.
RANK_FOLDERS = {1: "T8AzYUgNIAgTJ9Iz", 2: "lPzcDrjGDFbzuF2r", 3: "4fJHnYFqH0VjZXPv", 4: "7cyoYPoPYA01mErU", 5: "iDjm2cXyRAe4VZ35",
                6: "zDlxATrLAlbYA2Mv", 7: "xqlqJ6eeiNtWb7bq", 8: "IfuYSF4ymX48CZnm", 9: "fi1AYNmuCRvoT15b", 10: "x5tjEh1N0FHzbHkv"}
FOLDERS = {"Cantrip": "D7UVaOiB40wTR6kv", **{f"Rank {rank}": folder for rank, folder in RANK_FOLDERS.items()},
           "Focus": "wnvibtD90h7CStDC", "Focus / Magus": "v2AJ74hbni0AKAbI", "Focus / Necromancer": "T2rDJFX8zdRrFEq1",
           "Focus / Summoner": "RqMacIl0O4XRplUE", "Rituals": "6JF73zNhT8zbp1qC", "Impossible Spells": "Am3U4ScFSAZKIpj9"}
SAVES = ["none", "fortitude", "reflex", "will"]
AREAS = ["none", "burst", "emanation", "cone", "line", "cylinder", "square", "cube"]
MODELS = {"Iteration 2 – NB-LR (fast, main model)": 2, "Iteration 3 – NB-LR + transformer": 3}
DEFAULTS = {"name": "New Spell", "rank": 1, "description": "", "traits": [], "traditions": [], "rarity": "common",
            "save": "none", "basic": False, "area": "none", "area_value": 0, "damage": [], "sustained": False,
            "time": "2", "range": "", "duration": "", "target": "", "folder": "Rank 1", "base": {}, "editor_version": 0}

state = st.session_state
for key, value in DEFAULTS.items():
    state.setdefault(key, copy.deepcopy(value))


@st.cache_resource(show_spinner="Loading the models (first time ~1 min)...")
def load_predictor() -> SpellPredictor:
    return SpellPredictor()


def damage_formula(text: str, damage_type: str) -> str:
    match = re.search(rf"(\d+d\d+(?:\s*\+\s*\d+)?)\s+(?:persistent\s+)?{damage_type}", text)
    return match.group(1) if match else ""


def fill_from_model() -> None:
    if not state.description.strip():
        state.warning = "Write a description first."
        return
    fields = load_predictor().predict(state.rank, state.description, MODELS[state.model])
    text = clean_text(state.description)
    state.traits, state.traditions = fields["traits"], fields["traditions"]
    state.save = fields["save"][0]
    state.basic = state.save != "none" and f"basic {state.save}" in text
    state.area = fields["area"][0] if fields["area"][0] in AREAS else "none"
    size = re.search(rf"(\d+)-foot {state.area}", text)
    state.area_value = int(size.group(1)) if size else 0
    state.damage = [{"type": kind, "formula": damage_formula(text, kind)} for kind in fields["damage_types"]]
    state.sustained = bool(fields["sustained"])
    state.folder = "Cantrip" if "cantrip" in state.traits else "Focus" if "focus" in state.traits else f"Rank {state.rank}"
    state.editor_version += 1


def import_spell(spell: dict) -> None:
    system = spell.get("system", {})
    traits, save = system.get("traits", {}), (system.get("defense") or {}).get("save") or {}
    area, duration = system.get("area") or {}, system.get("duration") or {}
    state.update(
        base=spell, name=spell.get("name", ""), rank=min(max(int(system.get("level", {}).get("value", 1) or 1), 1), 10),
        description=system.get("description", {}).get("value", ""), traits=list(traits.get("value", [])),
        traditions=list(traits.get("traditions", [])), rarity=traits.get("rarity", "common"),
        save=save.get("statistic") or "none", basic=bool(save.get("basic")),
        area=area.get("type") if area.get("type") in AREAS else "none", area_value=int(area.get("value") or 0),
        damage=[{"type": entry.get("type", ""), "formula": entry.get("formula", "")} for entry in (system.get("damage") or {}).values()],
        sustained=bool(duration.get("sustained")), duration=duration.get("value", ""),
        time=(system.get("time") or {}).get("value", ""), range=(system.get("range") or {}).get("value", ""),
        target=(system.get("target") or {}).get("value", ""),
        folder=next((name for name, folder in FOLDERS.items() if folder == spell.get("folder")), state.folder),
    )
    state.editor_version += 1


def as_html(text: str) -> str:
    if re.search(r"<[a-z][^>]*>", text):
        return text
    return "\n".join(f"<p>{html.escape(part.strip())}</p>" for part in text.split("\n\n") if part.strip())


def to_foundry(damage_rows: list[dict]) -> dict:
    # Start from the imported file, so fields the app does not show are kept.
    spell = copy.deepcopy(state.base) or {"img": "systems/pf2e/icons/default-icons/spell.svg"}
    system = spell.setdefault("system", {})
    spell.update(name=state.name, type="spell", folder=FOLDERS[state.folder])
    def merged(key: str, values: dict) -> dict:
        return {**(system.get(key) or {}), **values}  # keep extra Foundry fields, e.g. area "details"

    old_defense = system.get("defense") or {}  # can also hold a passive defense like AC
    if state.save == "none":
        defense = {**old_defense, "save": None} if set(old_defense) - {"save"} else None
    else:
        defense = {**old_defense, "save": {**(old_defense.get("save") or {}), "statistic": state.save, "basic": state.basic}}
    old_damage = list((system.get("damage") or {}).items())  # Foundry keys damage entries by random ids
    system.update(
        description=merged("description", {"value": as_html(state.description)}),  # keeps e.g. the GM note
        level=merged("level", {"value": int(state.rank)}),
        traits=merged("traits", {"value": list(state.traits), "traditions": list(state.traditions), "rarity": state.rarity}),
        defense=defense,
        area=None if state.area == "none" else merged("area", {"type": state.area, "value": int(state.area_value)}),
        damage=dict(
            (old_damage[index][0], {**old_damage[index][1], "formula": row["formula"], "type": row["type"]}) if index < len(old_damage)
            else (secrets.token_hex(8), {"applyMod": False, "category": None, "formula": row["formula"], "kinds": ["damage"],
                                         "materials": [], "type": row["type"]})
            for index, row in enumerate(damage_rows)),
        duration=merged("duration", {"sustained": state.sustained, "value": state.duration}),
        time=merged("time", {"value": state.time}), range=merged("range", {"value": state.range}),
        target=merged("target", {"value": state.target}),
    )
    for key, value in {"cost": {"value": ""}, "counteraction": False, "requirements": "", "rules": [],
                       "publication": {"license": "ORC", "remaster": True, "title": "Homebrew"}}.items():
        system.setdefault(key, value)
    return spell


# ---------------------------------------------------------------- page
predictor = load_predictor()

with st.sidebar:
    st.header("🔮 PF2e Spell Tagger")
    st.radio("Model", list(MODELS), key="model")
    st.caption("Both models only read the rank and the description.")
    st.divider()
    uploaded = st.file_uploader("Import a Foundry spell (.json)", type="json")
    if uploaded and state.get("imported_file") != uploaded.file_id:
        state.imported_file = uploaded.file_id
        import_spell(json.load(uploaded))
        st.rerun()
    if st.button("Start a new spell", width="stretch"):
        for key, value in DEFAULTS.items():
            state[key] = copy.deepcopy(value)
        state.editor_version = state.get("editor_version", 0) + 1
        st.rerun()

st.title("PF2e Spell Tagger")
st.caption("Write a spell → the model fills in the Foundry fields → fix anything you like → download the JSON and import it in Foundry VTT.")

st.subheader("1. Your spell")
left, right = st.columns([3, 1])
left.text_input("Name", key="name")
right.number_input("Rank", min_value=1, max_value=10, step=1, key="rank")
st.text_area("Description", key="description", height=180,
             placeholder="A roaring blast of fire detonates at a spot you designate, dealing 6d6 fire damage...")
st.button("✨ Fill in the fields", type="primary", on_click=fill_from_model)
if warning := state.pop("warning", None):
    st.warning(warning)

st.subheader("2. Check and edit")
version = state.editor_version
labels = predictor.groups
col1, col2, col3 = st.columns(3)
state.traits = col1.multiselect("Traits", sorted(set(labels["traits"]) | set(state.traits)), default=state.traits,
                                accept_new_options=True, key=f"traits_{version}")
state.traditions = col1.multiselect("Traditions", labels["traditions"], default=[t for t in state.traditions if t in labels["traditions"]],
                                    key=f"traditions_{version}")
state.rarity = col1.selectbox("Rarity", ["common", "uncommon", "rare", "unique"],
                              index=["common", "uncommon", "rare", "unique"].index(state.rarity) if state.rarity in ["common", "uncommon", "rare", "unique"] else 0,
                              key=f"rarity_{version}")
state.save = col2.selectbox("Saving throw", SAVES, index=SAVES.index(state.save) if state.save in SAVES else 0, key=f"save_{version}")
state.basic = col2.checkbox("Basic save", value=state.basic, key=f"basic_{version}")
state.area = col2.selectbox("Area", AREAS, index=AREAS.index(state.area), key=f"area_{version}")
state.area_value = col2.number_input("Area size (feet)", min_value=0, step=5, value=int(state.area_value), key=f"area_value_{version}")
state.sustained = col2.checkbox("Sustained", value=state.sustained, key=f"sustained_{version}")
state.time = col3.text_input("Cast time (1, 2, 3, reaction, free, 1 minute...)", value=state.time, key=f"time_{version}")
state.range = col3.text_input("Range", value=state.range, key=f"range_{version}")
state.duration = col3.text_input("Duration", value=state.duration, key=f"duration_{version}")
state.target = col3.text_input("Targets", value=state.target, key=f"target_{version}")
state.folder = col3.selectbox("Foundry folder", list(FOLDERS), index=list(FOLDERS).index(state.folder), key=f"folder_{version}")
damage = st.data_editor(pd.DataFrame(state.damage, columns=["type", "formula"]), num_rows="dynamic", width="stretch",
                        column_config={"type": st.column_config.SelectboxColumn("Damage type", options=sorted(set(labels["damage_types"]) | {row["type"] for row in state.damage})),
                                       "formula": st.column_config.TextColumn("Formula, e.g. 6d6")},
                        key=f"damage_{version}")
# state.damage is only the starting table (set by the model or an import); the edited rows come back from the editor.
damage_rows = [row for row in damage.fillna("").to_dict("records") if row["type"]]

st.subheader("3. Download")
spell = to_foundry(damage_rows)
file_name = re.sub(r"[^a-z0-9]+", "-", state.name.lower()).strip("-") or "spell"
st.download_button("⬇️ Download Foundry JSON", json.dumps(spell, indent=4, ensure_ascii=False), file_name=f"{file_name}.json",
                   mime="application/json", type="primary")
st.caption("In Foundry: right-click a spell in the Items tab or a compendium → *Import Data* → choose this file.")
with st.expander("Show JSON"):
    st.json(spell)
