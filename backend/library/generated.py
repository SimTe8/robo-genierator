"""Generated objects: JSON specs (from the Gemini side quest) -> validated, normalised ObjectDefs.

Specs live in backend/library/generated/<id>.json and are merged into the object library at runtime
(`refresh_generated()`), so they appear in the UI and planner briefs like the hand-written objects.
No SDK here: this file only validates data. Gemini never writes code we execute.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

from library.objects import OBJECTS, OBJECTS_BY_ID, ChildBody, ObjectDef, Part, Physics, TaskRef

GENERATED_DIR = Path(__file__).resolve().parent / "generated"

# Presets that work on any object (no object-specific fixtures). Some need a named part or a button.
GENERIC_TASKS: dict[str, str] = {
    "pick_lift": "grasp it and hold it in the air",
    "precision_pinch": "lift with only thumb and index fingertips (small objects, < 4 cm)",
    "pick_by_part": "grasp it by its first named part (needs a named part, e.g. 'handle')",
    "rotate_in_hand": "lift and rotate it about the vertical axis in the hand (needs a visibly asymmetric object)",
    "reorient_to_target": "lift and match a target orientation (needs a visibly asymmetric object)",
    "flip_over": "turn it upside down on the table",
    "place_on_pad": "carry it onto a target pad",
    "place_in_bin": "put it into an 11 cm bin (object must fit)",
    "push_to_pad": "slide it onto a pad without lifting",
    "push_to_pose": "push it into a target pose (flat, asymmetric objects)",
    "tip_over": "knock it onto its side (tall objects)",
    "stand_upright": "it starts lying down; stand it up (tall objects)",
    "turn_part_to_hand": "rotate it so its first named part points at the hand (needs a named part)",
    "press_tool_button": "hold it and press its button (needs a button named 'button')",
}

Kind = Literal["box", "cylinder", "sphere", "capsule", "ellipsoid"]
N_SIZE = {"box": 3, "cylinder": 2, "sphere": 1, "capsule": 2, "ellipsoid": 3}


class GenPart(BaseModel):
    kind: Kind = Field(description="Primitive type")
    size: list[float] = Field(description="MuJoCo size in metres: box=[half_x,half_y,half_z], cylinder/capsule=[radius,half_length], sphere=[radius], ellipsoid=[rx,ry,rz]")
    pos: list[float] = Field(description="[x,y,z] centre in metres, object frame, z up")
    axis: Literal["x", "y", "z"] = Field("z", description="Long axis for cylinder/capsule")
    yaw_deg: float = Field(0.0, description="Extra rotation about z, degrees")
    name: str | None = Field(None, description="Optional part name, e.g. handle, lid, blade")
    rgba: list[float] | None = Field(None, description="Optional colour [r,g,b,a] 0..1")


class GenButton(BaseModel):
    name: str = Field(description="Button name; use 'button' for the main one")
    pos: list[float] = Field(description="[x,y,z] centre on the object's surface, object frame")
    half_size: list[float] = Field(description="Box half extents [x,y,z], metres (a few mm)")
    travel_mm: float = Field(4.0, description="Press depth in mm (2-8)")
    rgba: list[float] = Field(default_factory=lambda: [0.9, 0.2, 0.2, 1.0])


class GenPhysics(BaseModel):
    material: str
    fragility: Literal["low", "medium", "high"] = "low"
    deformability: Literal["rigid", "slightly", "very"] = "rigid"
    transparency: Literal["opaque", "translucent", "transparent"] = "opaque"
    contents: str = "none"
    can_contain_liquid: bool = False
    sealed: bool = False


class GeneratedAsset(BaseModel):
    name: str = Field(description="Short display name, e.g. 'TV remote'")
    description: str = Field(description="One sentence: what it is and what is interesting to manipulate")
    grasp_note: str = Field(description="One sentence of advice for a dexterous hand grasping it")
    mass_kg: float = Field(description="Realistic total mass in kg")
    rgba: list[float] = Field(description="Main colour [r,g,b,a] 0..1")
    yaw_symmetry_deg: float = Field(description="360 = looks different at every yaw, 180/90 = repeats, 0 = round about z")
    physics: GenPhysics
    parts: list[GenPart] = Field(description="1-12 primitives that together form the object; they must touch/overlap")
    buttons: list[GenButton] = Field(default_factory=list, description="Optional spring-loaded push buttons (0-4)")
    tasks: list[str] = Field(description="2-5 task ids from the allowed list that make sense for this object")

    @field_validator("rgba")
    @classmethod
    def _rgba(cls, v: list[float]) -> list[float]:
        v = (list(v) + [1.0] * 4)[:4]
        return [float(min(1, max(0, x))) for x in v]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:32] or "object"


def _part_corners(p: GenPart) -> np.ndarray:
    s = p.size
    if p.kind == "box":
        ext = np.array(s[:3])
    elif p.kind == "sphere":
        ext = np.array([s[0]] * 3)
    elif p.kind == "ellipsoid":
        ext = np.array(s[:3])
    else:  # cylinder / capsule along axis
        r, h = s[0], s[1] + (s[0] if p.kind == "capsule" else 0)
        ext = {"x": np.array([h, r, r]), "y": np.array([r, h, r]), "z": np.array([r, r, h])}[p.axis]
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * ext
    a = math.radians(p.yaw_deg)
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    return corners @ rz.T + np.array(p.pos)


def normalise(asset: GeneratedAsset) -> tuple[GeneratedAsset, list[str]]:
    """Fix sizes, drop bad tasks, re-centre on the bounding box. Raises ValueError if unusable."""
    warnings: list[str] = []
    if not 1 <= len(asset.parts) <= 12:
        raise ValueError(f"need 1-12 parts, got {len(asset.parts)}")
    for p in asset.parts:
        need = N_SIZE[p.kind]
        if len(p.size) < need:
            raise ValueError(f"part {p.name or p.kind}: size needs {need} numbers, got {p.size}")
        p.size = [float(min(0.15, max(0.001, v))) for v in p.size[:need]]
        p.pos = (list(p.pos) + [0, 0, 0])[:3]
    boxes = [_part_corners(p) for p in asset.parts]
    allc = np.vstack(boxes)
    lo, hi = allc.min(0), allc.max(0)
    centre, half = (lo + hi) / 2, (hi - lo) / 2
    if half.max() > 0.125:
        raise ValueError(f"object too large ({(2 * half * 100).round(1).tolist()} cm); keep it under 25 cm")
    if half.min() < 0.0015:
        raise ValueError("object is thinner than 3 mm")
    # every part should touch another one (AABB overlap with 2 mm slack)
    if len(boxes) > 1:
        aabb = [(b.min(0) - 0.002, b.max(0) + 0.002) for b in boxes]
        for i, (l1, h1) in enumerate(aabb):
            if not any(np.all(l1 <= h2) and np.all(l2 <= h1) for j, (l2, h2) in enumerate(aabb) if j != i):
                raise ValueError(f"part {asset.parts[i].name or i} floats free of the rest of the object")
    for p in asset.parts:
        p.pos = (np.array(p.pos) - centre).round(5).tolist()
    for b in asset.buttons:
        b.pos = (np.array((list(b.pos) + [0, 0, 0])[:3]) - centre).round(5).tolist()
        b.half_size = [float(min(0.02, max(0.002, v))) for v in (list(b.half_size) + [0.004] * 3)[:3]]
        b.travel_mm = float(min(8, max(2, b.travel_mm)))
    asset.buttons = asset.buttons[:4]
    asset.mass_kg = float(min(1.5, max(0.005, asset.mass_kg)))
    named = [p.name for p in asset.parts if p.name]
    keep = []
    for t in dict.fromkeys(asset.tasks):
        if t not in GENERIC_TASKS:
            warnings.append(f"dropped unknown task '{t}'")
        elif t in ("pick_by_part", "turn_part_to_hand") and not named:
            warnings.append(f"dropped '{t}': no named part")
        elif t == "press_tool_button" and not any(b.name == "button" for b in asset.buttons):
            warnings.append("dropped 'press_tool_button': no button named 'button'")
        else:
            keep.append(t)
    asset.tasks = keep or ["pick_lift"]
    if not keep:
        warnings.append("no valid tasks: defaulted to pick_lift")
    return asset, warnings


def to_object_def(asset: GeneratedAsset, obj_id: str) -> ObjectDef:
    allc = np.vstack([_part_corners(p) for p in asset.parts])
    half = tuple(float(v) for v in (allc.max(0) - allc.min(0)) / 2)
    parts = tuple(
        Part(p.kind, tuple(p.size), tuple(p.pos), p.axis, p.name, tuple(p.rgba) if p.rgba else None, p.yaw_deg)  # type: ignore[arg-type]
        for p in asset.parts
    )
    children = tuple(
        ChildBody(b.name, tuple(b.pos), (Part("box", tuple(b.half_size)),), range=(0.0, b.travel_mm / 1000), stiffness=150.0, rgba=tuple(b.rgba))  # type: ignore[arg-type]
        for b in asset.buttons
    )
    return ObjectDef(
        id=obj_id, name=asset.name, category="Generated", description=asset.description,
        half_extents=half, mass_kg=asset.mass_kg, rgba=tuple(asset.rgba),  # type: ignore[arg-type]
        parts=parts, tasks=tuple(TaskRef(t) for t in asset.tasks), yaw_symmetry_deg=asset.yaw_symmetry_deg,
        grasp_note=asset.grasp_note, physics=Physics(**asset.physics.model_dump()), children=children,
    )


def inline_schema() -> dict[str, Any]:
    """GeneratedAsset JSON schema with $refs inlined (simplest form for structured output)."""
    schema = GeneratedAsset.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            return {k: resolve(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


_loaded: dict[str, float] = {}


def refresh_generated() -> None:
    """Merge saved generated objects into the library (cheap; call before using OBJECTS)."""
    if not GENERATED_DIR.exists():
        return
    for path in sorted(GENERATED_DIR.glob("*.json")):
        mtime = path.stat().st_mtime
        if _loaded.get(path.stem) == mtime:
            continue
        try:
            data = json.loads(path.read_text())
            obj = to_object_def(GeneratedAsset.model_validate(data["asset"]), path.stem)
        except Exception:
            continue  # a broken file never takes the library down
        if path.stem in OBJECTS_BY_ID:
            OBJECTS[:] = [o for o in OBJECTS if o.id != path.stem]
        OBJECTS.append(obj)
        OBJECTS_BY_ID[obj.id] = obj
        _loaded[path.stem] = mtime
