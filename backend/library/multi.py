"""Scene layout: where the target goes, and the extra free bodies next to it (multi-object scenes).

Two sources of extra objects:
1. Multi-object task presets spawn their own (stack_tower: copies to stack, bin_all: copies to clear
   into the bin, sort_by_color: red/blue copies, stack_on_block: a base block, pick_from_clutter:
   different library objects packed around the target).
2. `clutter`: any library objects the user adds to the scene (works with every task).

Every extra is a free body named after its role (`stack_1`, `bin_item_2`, `sort_3`, `base_block`,
`other_1`, `clutter_1`, ...) so sim/check.py and the planner can address it.

Layout (`plan_layout`):
- no seed: the fixed default layout (target at the workspace origin, extras on the nearest free spots);
- `layout_seed=N`: "Shuffle layout" in the UI. Target and extras get random positions (and yaw where it
  is safe) drawn from a seeded RNG, so the same seed always rebuilds the same scene; the seed travels
  with the planner brief. Every object avoids the fixtures and every other object; if random sampling
  can't fit something it falls back to the nearest free spot. Ranges live in config.yaml -> layout.
- explicit `object_xy` / `object_yaw_deg` (the planner's own conditions) always win for the target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import mujoco
import numpy as np

from config import load_config
from library.objects import OBJECTS_BY_ID, ObjectDef, Part, Physics, add_parts
from library.tasks import TaskPreset

MAX_CLUTTER = 5
WORKSPACE_HALF = 0.165  # default-layout grid: extras within this square around the workspace origin (tables are >= 0.20)
TABLE_MARGIN = 0.03  # an object's footprint may reach this far beyond WORKSPACE_HALF / extras_range_m
SAMPLES = 600  # random tries per object before falling back to the nearest free spot
YAW_FIXED_GOALS = {"face_part"}  # a random start yaw could already satisfy the goal
DEFAULT_TARGET_XY = (0.0, 0.0)
PUSH_OFFSET = (0.0, -0.08)  # push_to_pose ghost outline relative to the target (same as scene.py)
LOOKALIKE_DY = 0.075  # look-alike cans sit this far either side of the target (scene.py `_add_special`)
GAP = 0.012  # clearance between neighbouring objects
RED = (0.90, 0.20, 0.18, 1.0)
BLUE = (0.18, 0.42, 0.95, 1.0)
TOWER_COLOURS = ((0.96, 0.65, 0.14, 1), (0.18, 0.70, 0.45, 1), (0.55, 0.35, 0.90, 1), (0.95, 0.95, 0.92, 1))
SORT_PADS = {"red": (-0.075, -0.105), "blue": (0.075, -0.105)}
SORT_PAD_HALF = 0.04

# Not in the library: the base for stack_on_block (8 x 8 x 4 cm wooden block, heavy enough to stay put).
BASE_BLOCK = ObjectDef(
    "base_block", "Base block", "Fixture", "8 x 8 x 4 cm wooden block to stack on.", (0.04, 0.04, 0.02), 0.35,
    (0.62, 0.46, 0.30, 1), (Part("box", (0.04, 0.04, 0.02)),), (), 90, "", Physics("wood"),
)
# Different everyday objects for pick_from_clutter, in order of preference (the target is skipped).
# Only things that rest stably: balls and eggs would roll into the target before the robot moves.
CLUTTER_POOL = ("die", "can", "l_block", "cube", "square_peg", "t_block", "pen")
# Fixture footprints that extras must avoid: name -> (x, y, radius) relative to the robot origin.
FIXTURE_FOOTPRINT = {
    "target_pad": (0.0, -0.10, 0.065), "bin": (0.0, -0.115, 0.085), "nail_board": (0.0, -0.08, 0.07),
    "lock": (0.0, -0.085, 0.06), "shape_board": (0.0, -0.09, 0.08), "screw_board": (0.0, -0.085, 0.07),
    "cargo": (-0.02, 0.09, 0.03), "bowl": (0.0, -0.10, 0.075),
    "sort_pads": (0.0, -0.105, 0.125), "base_block": (0.0, -0.10, 0.065),
}


@dataclass(frozen=True)
class Extra:
    name: str  # body name in the model
    obj: ObjectDef
    role: str  # stack | bin_item | sort | base | distractor | clutter
    xy: tuple[float, float]  # relative to the robot's workspace origin
    rgba: tuple[float, ...] | None = None
    colour: str | None = None  # sort_by_color: "red" / "blue"
    scale: float = 1.0
    yaw_deg: float = 0.0

    @property
    def radius(self) -> float:
        return footprint(self.obj, self.scale)


@dataclass(frozen=True)
class Layout:
    target_xy: tuple[float, float]  # relative to the robot's workspace origin
    target_yaw_deg: float
    extras: list[Extra]
    info: dict[str, Any]  # task-info additions (stack names, sort pads, ...)
    seed: int | None


def _part_radius(p: Part) -> float:
    """Conservative xy radius of one primitive around the object origin (any yaw)."""
    s = p.size
    if p.kind == "sphere":
        ex, ey = s[0], s[0]
    elif p.kind in ("cylinder", "capsule"):
        half_len = s[1] + (s[0] if p.kind == "capsule" else 0)
        ex, ey = (s[0], s[0]) if p.axis == "z" else (half_len, s[0])
    else:  # box, ellipsoid
        ex, ey = s[0], s[1]
    return math.hypot(p.pos[0], p.pos[1]) + math.hypot(ex, ey)


def footprint(obj: ObjectDef, scale: float = 1.0, lying: bool = False) -> float:
    """Radius of a circle (top view, any yaw) that contains the object, incl. an articulated object's base."""
    hx, hy, hz = obj.half_extents
    r = math.hypot(hx, hz if lying else hy)
    if obj.articulation is not None and obj.articulation.base_parts:
        r = max(r, *(_part_radius(p) for p in obj.articulation.base_parts))
    return r * scale


def can_be_extra(obj: ObjectDef) -> bool:
    """Plain rigid objects only (no articulated bodies, no objects with their own extra bodies)."""
    return obj.articulation is None and obj.special is None


Circle = tuple[float, float, float]  # (x, y, radius), relative to the workspace origin


def _clear(x: float, y: float, r: float, taken: list[Circle]) -> bool:
    return all(math.hypot(x - bx, y - by) >= r + br + GAP for bx, by, br in taken)


def _on_table(x: float, y: float, r: float, half: float) -> bool:
    return abs(x) + r <= half + TABLE_MARGIN and abs(y) + r <= half + TABLE_MARGIN


def free_slots(radii: list[float], blocked: list[Circle], near: float = 0.0,
               center: tuple[float, float] = (0.0, 0.0)) -> list[tuple[float, float]]:
    """Greedy, deterministic placement: for each radius, the free grid point closest to `center`
    (or closest to a ring of radius `near` around it, for clutter packed around the target)."""
    step = 0.005
    grid = np.arange(-WORKSPACE_HALF, WORKSPACE_HALF + 1e-9, step)
    pts = [(float(x), float(y)) for x in grid for y in grid]
    cx, cy = center
    pts.sort(key=lambda p: (round(abs(math.hypot(p[0] - cx, p[1] - cy) - near), 4), round(math.atan2(p[0] - cx, p[1] - cy), 4)))
    out: list[tuple[float, float]] = []
    taken = list(blocked)
    for r in radii:
        for x, y in pts:
            if _on_table(x, y, r, WORKSPACE_HALF) and _clear(x, y, r, taken):
                out.append((x, y))
                taken.append((x, y, r))
                break
        else:
            raise ValueError("not enough free space on the table for all objects; remove some")
    return out


def random_slots(radii: list[float], blocked: list[Circle], rng: np.random.Generator, half: float,
                 ring: tuple[float, float, float, float] | None = None) -> list[tuple[float, float]]:
    """Seeded placement: uniform on the table (or, with ring=(cx, cy, d_min, d_max), at a random distance
    in [d_min, d_max] from (cx, cy), measured between edges). Falls back to the nearest free spot."""
    out: list[tuple[float, float]] = []
    taken = list(blocked)
    for r in radii:
        for _ in range(SAMPLES):
            if ring:
                cx, cy, d0, d1 = ring
                ang, dist = rng.uniform(0, 2 * math.pi), rng.uniform(d0, d1) + r
                x, y = cx + dist * math.cos(ang), cy + dist * math.sin(ang)
            else:
                lim = max(0.0, half + TABLE_MARGIN - r)
                x, y = (float(v) for v in rng.uniform(-lim, lim, 2))
            if _on_table(x, y, r, half) and _clear(x, y, r, taken):
                break
        else:
            x, y = free_slots([r], taken, center=(ring[0], ring[1]) if ring else (0.0, 0.0))[0]
        out.append((float(x), float(y)))
        taken.append((x, y, r))
    return out


def fixture_blocks(task: TaskPreset) -> list[Circle]:
    """Fixed things the target and extras must avoid (fixtures, and the base block of stack_on_block)."""
    fx = task.fixture or ""
    out = [FIXTURE_FOOTPRINT[fx]] if fx in FIXTURE_FOOTPRINT else []
    if task.goal == "stack_on":
        out.append(FIXTURE_FOOTPRINT["base_block"])
    return out


def attached_to_target(obj: ObjectDef, task: TaskPreset, scale: float) -> list[Circle]:
    """Circles (relative to the target) that move with it: the object itself, look-alike cans, push ghost."""
    lying = task.initial_pose == "lying"
    r = footprint(obj, scale, lying)
    out: list[Circle] = [(0.0, 0.0, r)]
    if obj.special == "lookalike_cans":
        out += [(0.0, LOOKALIKE_DY, r), (0.0, -LOOKALIKE_DY, r)]
    if task.fixture == "ghost_table":
        out.append((PUSH_OFFSET[0], PUSH_OFFSET[1], r))
    return out


def plan_layout(task: TaskPreset, obj: ObjectDef, params: dict[str, float], clutter: list[str], scale: float, *,
                seed: int | None = None, attempt: int = 0, object_xy: tuple[float, float] | None = None,
                object_yaw_deg: float | None = None) -> Layout:
    """Where the target and every extra body go. Deterministic for (seed, attempt); seed None = default layout.
    `attempt` > 0 is a re-draw of the same seed (scene.py re-draws if something starts touching the robot)."""
    cfg = load_config().get("layout", {})
    rng = None if seed is None else np.random.default_rng(seed if attempt == 0 else [seed, attempt])
    extras_half = float(cfg.get("extras_range_m", WORKSPACE_HALF)) if rng else WORKSPACE_HALF
    fixed = fixture_blocks(task)
    attached = attached_to_target(obj, task, scale)

    # 1. the target
    txy = object_xy if object_xy is not None else DEFAULT_TARGET_XY
    if object_xy is None and rng is not None:
        lim = float(cfg.get("target_range_m", 0.08))
        for _ in range(SAMPLES):
            x, y = (float(v) for v in rng.uniform(-lim, lim, 2))
            if all(_on_table(x + dx, y + dy, r, extras_half) and _clear(x + dx, y + dy, r, fixed) for dx, dy, r in attached):
                txy = (x, y)
                break
    may_yaw = rng is not None and cfg.get("shuffle_yaw", True)
    yaw_ok = obj.articulation is None and task.goal not in YAW_FIXED_GOALS  # articulated bases don't rotate
    tyaw = object_yaw_deg if object_yaw_deg is not None else (float(rng.uniform(0, 360)) if may_yaw and yaw_ok else 0.0)  # type: ignore[union-attr]

    blocked = fixed + [(txy[0] + dx, txy[1] + dy, r) for dx, dy, r in attached]

    def place(radii: list[float], ring: tuple[float, float, float, float] | None = None, near: float = 0.0) -> list[tuple[float, float]]:
        if rng is not None:
            slots = random_slots(radii, blocked, rng, extras_half, ring)
        else:
            slots = free_slots(radii, blocked, near=near, center=txy if near else (0.0, 0.0))
        blocked.extend((x, y, r) for (x, y), r in zip(slots, radii))
        return slots

    def yaw() -> float:
        return float(rng.uniform(0, 360)) if may_yaw else 0.0  # type: ignore[union-attr]

    # 2. extras the task needs
    extras: list[Extra] = []
    info: dict[str, Any] = {}
    n = int(params.get("count", 0))
    r_self = footprint(obj, scale)
    if task.goal == "tower":
        slots = place([r_self] * (n - 1))
        extras += [Extra(f"stack_{i + 1}", obj, "stack", xy, TOWER_COLOURS[i % 4], scale=scale, yaw_deg=yaw()) for i, xy in enumerate(slots)]
        info["stack"] = [e.name for e in extras]
    elif task.goal == "all_inside":
        slots = place([r_self] * (n - 1))
        extras += [Extra(f"bin_item_{i + 1}", obj, "bin_item", xy, scale=scale, yaw_deg=yaw()) for i, xy in enumerate(slots)]
        info["bin_items"] = [e.name for e in extras]
    elif task.goal == "sort":
        colours = ["blue", "red", "blue", "red", "blue"][: n - 1]  # the target itself is red
        slots = place([r_self] * len(colours))
        extras += [Extra(f"sort_{i + 1}", obj, "sort", xy, RED if c == "red" else BLUE, c, scale, yaw())
                   for i, (xy, c) in enumerate(zip(slots, colours))]
        info["sort_items"] = {"target_object": "red", **{e.name: e.colour for e in extras}}
        info["sort_pads"] = {c: [x, y] for c, (x, y) in SORT_PADS.items()}  # relative; made absolute by scene.py
        info["sort_pad_half"] = SORT_PAD_HALF
        info["target_rgba"] = list(RED)
    elif task.goal == "stack_on":
        bx, by, _ = FIXTURE_FOOTPRINT["base_block"]  # the block is a fixture: it stays where the pad would be
        extras.append(Extra("base_block", BASE_BLOCK, "base", (bx, by)))
    elif task.goal == "clutter_pick":
        pool = [OBJECTS_BY_ID[i] for i in CLUTTER_POOL if i != obj.id and i in OBJECTS_BY_ID][:n]
        radii = [footprint(o) for o in pool]
        slots = place(radii, ring=(txy[0], txy[1], r_self + GAP, r_self + GAP + 0.04), near=r_self + 0.05)
        extras += [Extra(f"other_{i + 1}", o, "distractor", xy, yaw_deg=yaw()) for i, (o, xy) in enumerate(zip(pool, slots))]
        info["clutter_distractors"] = [e.name for e in extras]

    # 3. user-chosen clutter, after everything the task needs
    picks = [OBJECTS_BY_ID[c] for c in clutter[:MAX_CLUTTER]]
    if picks:
        slots = place([footprint(o) for o in picks])
        extras += [Extra(f"clutter_{i + 1}", o, "clutter", xy, yaw_deg=yaw()) for i, (o, xy) in enumerate(zip(picks, slots))]
    return Layout(txy, float(tyaw), extras, info, seed)


def recolour_main(parts: tuple[Part, ...]) -> tuple[Part, ...]:
    """Drop the main part's own colour so a new body colour applies; marks and caps keep theirs."""
    return tuple(replace(p, rgba=None) if i == 0 else p for i, p in enumerate(parts))


def add_extra(spec: mujoco.MjSpec, e: Extra, origin: tuple[float, float], top_z: float) -> list[float]:
    """Add one free body; returns its start position (world frame)."""
    b = spec.worldbody.add_body()
    b.name = e.name
    hz = e.obj.half_extents[2] * e.scale
    b.pos = [origin[0] + e.xy[0], origin[1] + e.xy[1], top_z + hz + 0.001]
    if e.yaw_deg:
        a = math.radians(e.yaw_deg) / 2
        b.quat = [math.cos(a), 0.0, 0.0, math.sin(a)]
    j = b.add_joint()
    j.name = f"{e.name}_freejoint"
    j.type = mujoco.mjtJoint.mjJNT_FREE
    parts = recolour_main(e.obj.parts) if e.rgba is not None else e.obj.parts
    add_parts(b, parts, e.rgba or e.obj.rgba, e.obj.mass_kg, e.scale, name_prefix=e.name)
    return [float(v) for v in b.pos]


def describe_extras(extras: list[Extra]) -> list[dict[str, Any]]:
    return [{"body": e.name, "object_id": e.obj.id, "name": e.obj.name, "role": e.role, **({"colour": e.colour} if e.colour else {}),
             "yaw_deg": round(e.yaw_deg, 1),
             "size_cm": [round(2 * h * e.scale * 100, 1) for h in e.obj.half_extents], "mass_kg": e.obj.mass_kg} for e in extras]
