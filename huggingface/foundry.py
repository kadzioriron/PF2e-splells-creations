"""Foundry VTT spell JSON: model output -> editable fields -> JSON, and JSON -> fields (import)."""
import copy
import html
import re
import secrets

from predictor import clean_text

# Folder ids from the Foundry export (data/spellsdata/_folders.json): the spell lands in the right folder on import.
RANK_FOLDERS = {1: "T8AzYUgNIAgTJ9Iz", 2: "lPzcDrjGDFbzuF2r", 3: "4fJHnYFqH0VjZXPv", 4: "7cyoYPoPYA01mErU", 5: "iDjm2cXyRAe4VZ35",
                6: "zDlxATrLAlbYA2Mv", 7: "xqlqJ6eeiNtWb7bq", 8: "IfuYSF4ymX48CZnm", 9: "fi1AYNmuCRvoT15b", 10: "x5tjEh1N0FHzbHkv"}
FOLDERS = {"Cantrip": "D7UVaOiB40wTR6kv", **{f"Rank {rank}": folder for rank, folder in RANK_FOLDERS.items()},
           "Focus": "wnvibtD90h7CStDC", "Focus / Magus": "v2AJ74hbni0AKAbI", "Focus / Necromancer": "T2rDJFX8zdRrFEq1",
           "Focus / Summoner": "RqMacIl0O4XRplUE", "Rituals": "6JF73zNhT8zbp1qC", "Impossible Spells": "Am3U4ScFSAZKIpj9"}
SAVES = ["none", "fortitude", "reflex", "will"]
AREAS = ["none", "burst", "emanation", "cone", "line", "cylinder", "square", "cube"]
RARITIES = ["common", "uncommon", "rare", "unique"]
EMPTY = {"name": "New Spell", "rank": 1, "description": "", "traits": [], "traditions": [], "rarity": "common",
         "save": "none", "basic": False, "area": "none", "area_value": 0, "damage": [], "sustained": False,
         "time": "2", "range": "", "duration": "", "target": "", "folder": "Rank 1"}


def from_prediction(fields: dict, prediction: dict[str, list[str]]) -> dict:
    """Puts the model output into the fields; formulas and sizes are read from the description."""
    text = clean_text(fields["description"])
    save, area = prediction["save"][0], prediction["area"][0]
    area = area if area in AREAS else "none"
    size = re.search(rf"(\d+)-foot {area}", text)
    damage = []
    for kind in prediction["damage_types"]:
        formula = re.search(rf"(\d+d\d+(?:\s*\+\s*\d+)?)\s+(?:persistent\s+)?{kind}", text)
        damage.append({"type": kind, "formula": formula.group(1) if formula else ""})
    traits = prediction["traits"]
    return {**fields, "traits": traits, "traditions": prediction["traditions"], "save": save,
            "basic": save != "none" and f"basic {save}" in text, "area": area, "area_value": int(size.group(1)) if size else 0,
            "damage": damage, "sustained": bool(prediction["sustained"]),
            "folder": "Cantrip" if "cantrip" in traits else "Focus" if "focus" in traits else f"Rank {fields['rank']}"}


def read_spell(spell: dict) -> dict:
    """Foundry JSON -> fields (import)."""
    system = spell.get("system", {})
    traits, save = system.get("traits", {}), (system.get("defense") or {}).get("save") or {}
    area, duration = system.get("area") or {}, system.get("duration") or {}
    return {
        "name": spell.get("name", ""), "rank": min(max(int(system.get("level", {}).get("value", 1) or 1), 1), 10),
        "description": system.get("description", {}).get("value", ""), "traits": list(traits.get("value", [])),
        "traditions": list(traits.get("traditions", [])), "rarity": traits.get("rarity", "common"),
        "save": save.get("statistic") or "none", "basic": bool(save.get("basic")),
        "area": area.get("type") if area.get("type") in AREAS else "none", "area_value": int(area.get("value") or 0),
        "damage": [{"type": entry.get("type", ""), "formula": entry.get("formula", "")} for entry in (system.get("damage") or {}).values()],
        "sustained": bool(duration.get("sustained")), "duration": duration.get("value", ""),
        "time": (system.get("time") or {}).get("value", ""), "range": (system.get("range") or {}).get("value", ""),
        "target": (system.get("target") or {}).get("value", ""),
        "folder": next((name for name, folder in FOLDERS.items() if folder == spell.get("folder")), EMPTY["folder"]),
    }


def as_html(text: str) -> str:
    if re.search(r"<[a-z][^>]*>", text):
        return text
    return "\n".join(f"<p>{html.escape(part.strip())}</p>" for part in text.split("\n\n") if part.strip())


def build_spell(fields: dict, base: dict | None = None) -> dict:
    """Fields -> Foundry JSON. Starts from the imported file (base), so fields the app does not show are kept."""
    spell = copy.deepcopy(base) if base else {"img": "systems/pf2e/icons/default-icons/spell.svg"}
    system = spell.setdefault("system", {})
    spell.update(name=fields["name"], type="spell", folder=FOLDERS[fields["folder"]])

    def merged(key: str, values: dict) -> dict:
        return {**(system.get(key) or {}), **values}  # keep extra Foundry fields, e.g. area "details"

    old_defense = system.get("defense") or {}  # can also hold a passive defense like AC
    if fields["save"] == "none":
        defense = {**old_defense, "save": None} if set(old_defense) - {"save"} else None
    else:
        defense = {**old_defense, "save": {**(old_defense.get("save") or {}), "statistic": fields["save"], "basic": fields["basic"]}}
    old_damage = list((system.get("damage") or {}).items())  # Foundry keys damage entries by random ids
    system.update(
        description=merged("description", {"value": as_html(fields["description"])}),
        level=merged("level", {"value": int(fields["rank"])}),
        traits=merged("traits", {"value": list(fields["traits"]), "traditions": list(fields["traditions"]), "rarity": fields["rarity"]}),
        defense=defense,
        area=None if fields["area"] == "none" else merged("area", {"type": fields["area"], "value": int(fields["area_value"])}),
        damage=dict(
            (old_damage[index][0], {**old_damage[index][1], "formula": row["formula"], "type": row["type"]}) if index < len(old_damage)
            else (secrets.token_hex(8), {"applyMod": False, "category": None, "formula": row["formula"], "kinds": ["damage"],
                                         "materials": [], "type": row["type"]})
            for index, row in enumerate(fields["damage"])),
        duration=merged("duration", {"sustained": fields["sustained"], "value": fields["duration"]}),
        time=merged("time", {"value": fields["time"]}), range=merged("range", {"value": fields["range"]}),
        target=merged("target", {"value": fields["target"]}),
    )
    for key, value in {"cost": {"value": ""}, "counteraction": False, "requirements": "", "rules": [],
                       "publication": {"license": "ORC", "remaster": True, "title": "Homebrew"}}.items():
        system.setdefault(key, value)
    return spell
