"""Object library for dexterous manipulation, built from MuJoCo primitives.

Built from primitives on purpose: written today, no third-party meshes, so no licensing questions.
Task ideas follow standard dexterous benchmarks (in-hand reorientation, valve turning, tool use, insertion)
and PG-VLM's physically grounded concepts (https://iliad.stanford.edu/pg-vlm/): every object carries
material / fragility / deformability / transparency / contents / can-contain-liquid / sealed annotations.

Conventions
  * Free objects: body `target_object` with a freejoint; origin = centre of the bounding box in the
    natural resting pose; z is up in that pose.
  * Articulated objects: static base geoms + body `target_object` on a hinge/slide joint
    `target_object_joint`, or a static `target_object` with jointed children (buttons).
  * Children (buttons, triggers): bodies `target_object_<name>` with joint `<name>_joint`.
  * First geom is `target_object_geom` (the executor tracks contacts on it); named parts are
    `target_object_<part>` so tasks can refer to them (e.g. a mug's handle).
  * Masses are spread over the geoms by volume so the total equals `mass_kg`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

G = mujoco.mjtGeom


@dataclass(frozen=True)
class Part:
    kind: str  # box | cylinder | sphere | capsule | ellipsoid
    size: tuple[float, ...]
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: str = "z"  # cylinders/capsules: the axis they run along
    name: str | None = None
    rgba: tuple[float, float, float, float] | None = None  # override (markers, caps, …)
    yaw_deg: float = 0.0  # extra rotation about the object's z axis (e.g. valve spokes at 120°)
    contype: int = 1
    conaffinity: int = 1

    def volume(self) -> float:
        s = self.size
        if self.kind == "box":
            return 8 * s[0] * s[1] * s[2]
        if self.kind == "cylinder":
            return math.pi * s[0] ** 2 * 2 * s[1]
        if self.kind == "sphere":
            return 4 / 3 * math.pi * s[0] ** 3
        if self.kind == "capsule":
            return math.pi * s[0] ** 2 * 2 * s[1] + 4 / 3 * math.pi * s[0] ** 3
        return 4 / 3 * math.pi * s[0] * s[1] * s[2]  # ellipsoid


@dataclass(frozen=True)
class Articulation:
    """Fixed-in-place object. `pivot_z` = height of the `target_object` origin above the table."""

    base_parts: tuple[Part, ...]
    pivot_z: float
    kind: str = "hinge"  # hinge | slide | static
    axis: tuple[float, float, float] = (0, 0, 1)
    range: tuple[float, float] | None = None
    damping: float = 0.02
    frictionloss: float = 0.01
    base_rgba: tuple[float, float, float, float] = (0.35, 0.37, 0.42, 1)


@dataclass(frozen=True)
class ChildBody:
    """A jointed sub-part (button, trigger). Spring-loaded back to 0 when `stiffness` > 0."""

    name: str
    pos: tuple[float, float, float]
    parts: tuple[Part, ...]
    kind: str = "slide"
    axis: tuple[float, float, float] = (0, 0, -1)
    range: tuple[float, float] = (0.0, 0.005)
    stiffness: float = 300.0
    damping: float = 1.0
    rgba: tuple[float, float, float, float] = (0.9, 0.2, 0.2, 1)


@dataclass(frozen=True)
class Physics:
    """PG-VLM physical concepts (ground truth here, so the planner/critic can be tested against them)."""

    material: str
    fragility: str = "low"  # low | medium | high
    deformability: str = "rigid"  # rigid | slightly | very
    transparency: str = "opaque"  # opaque | translucent | transparent
    contents: str = "none"
    can_contain_liquid: bool = False
    sealed: bool = False


@dataclass(frozen=True)
class TaskRef:
    task_id: str
    params: dict[str, float] = field(default_factory=dict)  # per-object defaults/overrides


@dataclass(frozen=True)
class ObjectDef:
    id: str
    name: str
    category: str
    description: str
    half_extents: tuple[float, float, float]  # bounding box in the natural pose (moving part for articulated)
    mass_kg: float
    rgba: tuple[float, float, float, float]
    parts: tuple[Part, ...]
    tasks: tuple[TaskRef, ...]
    yaw_symmetry_deg: float  # 360 = none, 180/90 = repeats, 0 = round about z (yaw unobservable)
    grasp_note: str
    physics: Physics
    articulation: Articulation | None = None
    children: tuple[ChildBody, ...] = ()
    special: str | None = None  # extra bodies built in scene.py (beads, look-alike cans)

    @property
    def dims_cm(self) -> list[float]:
        return [round(2 * h * 100, 1) for h in self.half_extents]

    @property
    def part_names(self) -> list[str]:
        names: list[str] = []
        for p in self.parts:  # skip markers and sub-pieces (handle_low, spoke_2, …)
            if p.name and "mark" not in p.name and p.name != "pointer" and not any(p.name.startswith(n + "_") for n in names):
                names.append(p.name)
        return names + [c.name for c in self.children]


_QUAT_ALONG = {"z": [1, 0, 0, 0], "x": [0.7071068, 0, 0.7071068, 0], "y": [0.7071068, -0.7071068, 0, 0]}
_KIND = {"box": G.mjGEOM_BOX, "cylinder": G.mjGEOM_CYLINDER, "sphere": G.mjGEOM_SPHERE,
         "capsule": G.mjGEOM_CAPSULE, "ellipsoid": G.mjGEOM_ELLIPSOID}


def _part_quat(p: Part) -> list[float]:
    base = _QUAT_ALONG[p.axis]
    if not p.yaw_deg:
        return base
    h = math.radians(p.yaw_deg) / 2
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.array([math.cos(h), 0.0, 0.0, math.sin(h)]), np.array(base, dtype=float))
    return out.tolist()


def add_parts(
    body: mujoco.MjsBody,
    parts: tuple[Part, ...],
    rgba: tuple[float, ...] | list[float],
    total_mass: float,
    scale: float = 1.0,
    visual_only: bool = False,
    name_prefix: str | None = "target_object",
) -> None:
    """Add geoms for `parts` to `body`. visual_only=True gives a non-colliding ghost (target marker)."""
    total_vol = sum(p.volume() for p in parts) or 1.0
    for i, p in enumerate(parts):
        g = body.add_geom()
        if name_prefix:
            g.name = f"{name_prefix}_geom" if i == 0 else f"{name_prefix}_{p.name or f'part{i}'}"
        g.type = _KIND[p.kind]
        size = [v * scale for v in p.size]
        g.size = size + [0.0] * (3 - len(size))
        g.pos = [v * scale for v in p.pos]
        g.quat = _part_quat(p)
        g.rgba = list(p.rgba or rgba)
        if visual_only:
            g.contype = 0
            g.conaffinity = 0
            g.group = 1
            g.rgba = [0.15, 0.85, 0.35, 0.35]
            g.mass = 0.0
        else:
            g.mass = total_mass * scale**3 * p.volume() / total_vol if total_mass > 0 else 0.0
            g.condim = 4
            g.friction = [1.0, 0.01, 0.0005]
            g.contype = p.contype
            g.conaffinity = p.conaffinity


def T(task_id: str, **params: float) -> TaskRef:
    return TaskRef(task_id, params)


METAL, PLASTIC, WOOD = "metal", "plastic", "wood"
B = "box"

OBJECTS: list[ObjectDef] = [
    # ---- In-hand reorientation ---------------------------------------------
    ObjectDef(
        "cube", "Cube", "In-hand", "6 cm block with a marked face, the classic in-hand reorientation object.",
        (0.03, 0.03, 0.03), 0.10, (0.92, 0.26, 0.21, 1),
        (Part(B, (0.03, 0.03, 0.03)), Part(B, (0.012, 0.012, 0.0015), (0, 0, 0.0305), name="top_mark", rgba=(1, 1, 1, 1))),
        (T("pick_lift"), T("rotate_in_hand", angle_deg=90), T("reorient_to_target", axis=0, angle_deg=90),
         T("flip_over"), T("place_on_pad"), T("place_in_bin")),
        90, "Power grasp with all fingers, or cradle in the palm for in-hand rotation.", Physics(WOOD),
    ),
    ObjectDef(
        "die", "Small die", "In-hand", "3 cm die with a marked face: precision fingertip work.",
        (0.015, 0.015, 0.015), 0.03, (0.95, 0.95, 0.92, 1),
        (Part(B, (0.015, 0.015, 0.015)), Part(B, (0.006, 0.006, 0.001), (0, 0, 0.0155), name="top_mark", rgba=(0.1, 0.1, 0.1, 1))),
        (T("precision_pinch"), T("rotate_in_hand", angle_deg=90), T("reorient_to_target", axis=1, angle_deg=90), T("place_on_pad")),
        90, "Fingertip pinch between thumb and index.", Physics(PLASTIC),
    ),
    ObjectDef(
        "egg", "Egg", "In-hand", "Egg-shaped ellipsoid, 5 × 7 cm. Fragile.",
        (0.025, 0.025, 0.035), 0.06, (0.96, 0.90, 0.78, 1),
        (Part("ellipsoid", (0.025, 0.025, 0.035)),),
        (T("pick_lift"), T("reorient_to_target", axis=0, angle_deg=90), T("place_on_pad")),
        0, "Curved everywhere: enclose it with the fingers, don't squeeze.",
        Physics("shell", fragility="high", contents="liquid (egg)", sealed=True),
    ),
    ObjectDef(
        "pen", "Pen", "In-hand", "14 cm pen lying on the table (1.2 cm thick).",
        (0.07, 0.006, 0.006), 0.02, (0.15, 0.30, 0.85, 1),
        (Part("cylinder", (0.006, 0.062), (-0.008, 0, 0), axis="x"), Part("cylinder", (0.0065, 0.008), (0.062, 0, 0), axis="x", name="cap", rgba=(0.9, 0.9, 0.9, 1))),
        (T("pick_lift"), T("rotate_in_hand", angle_deg=90), T("reorient_to_target", axis=1, angle_deg=90), T("place_in_bin")),
        360, "Thin and flat on the table: sweep fingertips under it or pinch it near the middle.", Physics(PLASTIC, contents="ink"),
    ),
    ObjectDef(
        "t_block", "T-block", "In-hand", "Flat T-shaped piece: asymmetric, so every rotation is visible.",
        (0.05, 0.05, 0.015), 0.10, (0.96, 0.65, 0.14, 1),
        (Part(B, (0.05, 0.0125, 0.015), (0, 0.0375, 0)), Part(B, (0.0125, 0.0375, 0.015), (0, -0.0125, 0), name="stem")),
        (T("rotate_in_hand", angle_deg=90), T("push_to_pose", yaw_deg=90), T("pick_lift")),
        360, "Grasp the 2.5 cm stem or push the edges.", Physics(PLASTIC),
    ),
    ObjectDef(
        "l_block", "L-block", "In-hand", "Flat L-shaped piece.",
        (0.045, 0.045, 0.015), 0.09, (0.10, 0.68, 0.62, 1),
        (Part(B, (0.045, 0.0125, 0.015), (0, -0.0325, 0)), Part(B, (0.0125, 0.0325, 0.015), (0.0325, 0.0125, 0), name="short_arm")),
        (T("rotate_in_hand", angle_deg=180), T("push_to_pose", yaw_deg=180), T("pick_lift")),
        360, "Grasp either 2.5 cm arm.", Physics(PLASTIC),
    ),
    # ---- Insertion -----------------------------------------------------------
    ObjectDef(
        "key", "Key", "Insertion", "3 mm-thin metal key lying flat. Insert it in the lock and turn it.",
        (0.03, 0.012, 0.0015), 0.015, (0.85, 0.72, 0.30, 1),
        (Part(B, (0.012, 0.012, 0.0015), (-0.018, 0, 0), name="bow", contype=3, conaffinity=3),
         Part(B, (0.018, 0.004, 0.0012), (0.012, 0, 0), name="blade", contype=3, conaffinity=3)),
        (T("insert_key"), T("unlock_with_key"), T("pick_lift", lift_cm=5)),
        360, "Very thin: slide it to the table edge or pinch the bow; it must be stood upright to go in.", Physics(METAL),
    ),
    ObjectDef(
        "round_peg", "Round peg", "Insertion", "2.4 cm round peg, 8 cm tall, for the shape sorter.",
        (0.012, 0.012, 0.04), 0.03, (0.35, 0.45, 0.95, 1),
        (Part("cylinder", (0.012, 0.04)),),
        (T("insert_matching_hole"), T("pick_lift"), T("tip_over")),
        0, "Grasp near the top so the bottom can go into the hole.", Physics(WOOD),
    ),
    ObjectDef(
        "square_peg", "Square peg", "Insertion", "2.4 cm square peg: only fits the square hole, and only when aligned.",
        (0.012, 0.012, 0.04), 0.035, (0.92, 0.26, 0.21, 1),
        (Part(B, (0.012, 0.012, 0.04)),),
        (T("insert_matching_hole"), T("pick_lift"), T("rotate_in_hand", angle_deg=45)),
        90, "Grasp near the top; align its faces with the hole before inserting.", Physics(WOOD),
    ),
    # ---- Tools ---------------------------------------------------------------
    ObjectDef(
        "power_screwdriver", "Power screwdriver", "Tools", "Cordless screwdriver with a spring-loaded button on top. Hold it, press the button, drive the screw.",
        (0.065, 0.016, 0.016), 0.25, (0.95, 0.55, 0.10, 1),
        (Part("cylinder", (0.016, 0.045), (-0.02, 0, 0), axis="x", name="grip"),
         Part("cylinder", (0.008, 0.008), (0.033, 0, 0), axis="x", name="chuck", rgba=(0.2, 0.2, 0.22, 1)),
         Part("cylinder", (0.0025, 0.012), (0.053, 0, 0), axis="x", name="bit", rgba=(0.75, 0.75, 0.8, 1))),
        (T("drive_screw"), T("press_tool_button"), T("pick_by_part")),
        360, "Wrap the 3.2 cm body; press the button on top with the thumb or index finger.", Physics(PLASTIC, contents="battery, motor"),
        children=(ChildBody("button", (0.0, 0, 0.016), (Part(B, (0.007, 0.005, 0.003)),), range=(0.0, 0.004), stiffness=150.0),),
    ),
    ObjectDef(
        "hammer", "Hammer", "Tools", "15 cm hammer lying on its side.",
        (0.075, 0.03, 0.012), 0.20, (0.45, 0.30, 0.18, 1),
        (Part("cylinder", (0.009, 0.065), (-0.01, 0, 0), axis="x", name="handle"),
         Part(B, (0.012, 0.03, 0.012), (0.063, 0, 0), name="head", rgba=(0.5, 0.52, 0.56, 1))),
        (T("pick_by_part"), T("hammer_nail")),
        360, "Grasp the 1.8 cm handle; strike with the head.", Physics("wood + metal"),
    ),
    ObjectDef(
        "screwdriver", "Screwdriver", "Tools", "14 cm manual screwdriver lying on its side.",
        (0.07, 0.012, 0.012), 0.06, (0.85, 0.25, 0.20, 1),
        (Part("cylinder", (0.012, 0.035), (-0.035, 0, 0), axis="x", name="handle"),
         Part("cylinder", (0.003, 0.035), (0.035, 0, 0), axis="x", name="shaft", rgba=(0.75, 0.75, 0.8, 1))),
        (T("pick_by_part"), T("reorient_to_target", axis=1, angle_deg=90), T("place_in_bin")),
        360, "Grasp the 2.4 cm handle; the shaft is too thin to hold.", Physics("plastic + metal"),
    ),
    # ---- Containers & physical reasoning (PG-VLM) ----------------------------------
    ObjectDef(
        "cup_of_beads", "Cup of beads", "Physical reasoning", "Open cup holding 10 beads. Pour them into the bowl without spilling.",
        (0.03, 0.03, 0.04), 0.06, (0.70, 0.85, 0.95, 0.55),
        tuple([Part(B, (0.03, 0.03, 0.003), (0, 0, -0.037))]
              + [Part(B, (0.003, 0.0125, 0.037), (0.0275 * math.cos(a), 0.0275 * math.sin(a), 0.003), name=f"wall_{i}" if i else "wall", yaw_deg=math.degrees(a))
                 for i, a in enumerate(np.linspace(0, 2 * math.pi, 8, endpoint=False))]),
        (T("pour_beads"), T("pick_lift", lift_cm=8)),
        0, "Grasp the cup from the side, keep it upright until it is over the bowl, then tilt.",
        Physics(PLASTIC, transparency="translucent", contents="10 beads", can_contain_liquid=True),
        special="beads",
    ),
    ObjectDef(
        "mystery_cans", "Three look-alike cans", "Physical reasoning", "Three identical-looking cans; only one is full. Find the heaviest by touch.",
        (0.025, 0.025, 0.045), 0.40, (0.55, 0.57, 0.62, 1),
        (Part("cylinder", (0.025, 0.045)),),
        (T("lift_heaviest"),),
        0, "Each can is 5 cm wide; lift or push them to compare weight.",
        Physics(METAL, contents="unknown (one full, two empty)", can_contain_liquid=True, sealed=True),
        special="lookalike_cans",
    ),
    ObjectDef(
        "mug", "Mug", "Physical reasoning", "Ceramic mug with a side handle (handle on +x).",
        (0.05, 0.032, 0.04), 0.20, (0.95, 0.95, 0.95, 1),
        (Part("cylinder", (0.032, 0.04), (-0.008, 0, 0)),
         Part(B, (0.009, 0.006, 0.004), (0.031, 0, 0.022), name="handle"),
         Part(B, (0.009, 0.006, 0.004), (0.031, 0, -0.022), name="handle_low"),
         Part(B, (0.004, 0.006, 0.026), (0.040, 0, 0), name="handle_bar")),
        (T("pick_by_part"), T("turn_part_to_hand"), T("place_on_pad")),
        360, "Hook fingers through the handle, or wrap the 6.4 cm body.",
        Physics("ceramic", fragility="high", can_contain_liquid=True),
    ),
    ObjectDef(
        "bottle", "Bottle", "Physical reasoning", "15 cm sealed bottle with a narrow neck.",
        (0.025, 0.025, 0.075), 0.20, (0.20, 0.60, 0.35, 0.8),
        (Part("cylinder", (0.025, 0.05), (0, 0, -0.025)),
         Part("cylinder", (0.011, 0.018), (0, 0, 0.043), name="neck"),
         Part("cylinder", (0.013, 0.007), (0, 0, 0.068), name="cap", rgba=(0.9, 0.9, 0.9, 1))),
        (T("pick_by_part"), T("stand_upright"), T("tip_over")),
        0, "Grasp the 2.2 cm neck with the fingertips, or wrap the body.",
        Physics("glass", fragility="high", transparency="translucent", contents="water", can_contain_liquid=True, sealed=True),
    ),
    ObjectDef(
        "ball", "Ball", "Physical reasoning", "6 cm soft ball. Rolls when pushed.",
        (0.03, 0.03, 0.03), 0.06, (0.85, 0.85, 0.20, 1),
        (Part("sphere", (0.03,)),),
        (T("pick_lift"), T("place_in_bin"), T("push_to_pad")),
        0, "Cage it with all fingers; it squeezes out of a two-finger grasp.", Physics("rubber", deformability="very"),
    ),
    ObjectDef(
        "can", "Can", "Physical reasoning", "Soda-can sized cylinder, 6 × 11 cm.",
        (0.03, 0.03, 0.055), 0.15, (0.80, 0.10, 0.12, 1),
        (Part("cylinder", (0.03, 0.055)),),
        (T("pick_lift"), T("tip_over"), T("stand_upright"), T("place_on_pad")),
        0, "Wrap grasp around the body.", Physics("aluminium", deformability="slightly", contents="soda", can_contain_liquid=True, sealed=True),
    ),
    ObjectDef(
        "card", "Card", "Physical reasoning", "Thin 6 × 9 cm card, 4 mm thick, flat on the table.",
        (0.03, 0.045, 0.002), 0.01, (0.30, 0.30, 0.34, 1),
        (Part(B, (0.03, 0.045, 0.002)),),
        (T("pick_lift", lift_cm=5), T("flip_over"), T("push_to_pad")),
        180, "Too flat to grasp directly: slide it to the edge, or lift one side with a fingertip.", Physics("cardboard", deformability="slightly"),
    ),
    # ---- Articulated (fixed in place) ---------------------------------------------
    ObjectDef(
        "button_panel", "Button panel", "Articulated", "Panel with red, green and blue spring-loaded buttons.",
        (0.06, 0.025, 0.01), 0.0, (0.25, 0.27, 0.30, 1),
        (Part(B, (0.06, 0.025, 0.01)),),
        (T("press_button", button=2), T("press_two_buttons")),
        360, "Press straight down with a fingertip; two fingers for two buttons.", Physics(PLASTIC),
        articulation=Articulation((), pivot_z=0.01, kind="static"),
        children=tuple(ChildBody(f"button{i + 1}", (x, 0, 0.012), (Part("cylinder", (0.009, 0.004)),), range=(0.0, 0.006), stiffness=120.0, rgba=c)
                       for i, (x, c) in enumerate([(-0.035, (0.9, 0.15, 0.15, 1)), (0.0, (0.15, 0.8, 0.3, 1)), (0.035, (0.2, 0.4, 0.95, 1))])),
    ),
    ObjectDef(
        "drawer", "Drawer", "Articulated", "Small cabinet drawer with a handle, opening toward the robot (−x).",
        (0.05, 0.045, 0.02), 0.10, (0.70, 0.55, 0.38, 1),
        (Part(B, (0.045, 0.04, 0.003), (0, 0, -0.017), name="floor"),
         Part(B, (0.003, 0.045, 0.02), (-0.048, 0, 0), name="front"),
         Part(B, (0.003, 0.04, 0.017), (0.042, 0, 0.0), name="back"),
         Part(B, (0.045, 0.003, 0.017), (0, 0.042, 0), name="side"),
         Part(B, (0.045, 0.003, 0.017), (0, -0.042, 0), name="side_2"),
         Part("capsule", (0.004, 0.015), (-0.057, 0, 0.004), axis="y", name="handle", rgba=(0.6, 0.6, 0.65, 1))),
        (T("open_drawer"), T("open_drawer_and_place")),
        360, "Hook a finger behind the handle and pull toward the robot.", Physics(WOOD, contents="empty"),
        articulation=Articulation(
            (Part(B, (0.055, 0.053, 0.003), (0.005, 0, 0.024)), Part(B, (0.055, 0.003, 0.024), (0.005, 0.05, 0.0)),
             Part(B, (0.055, 0.003, 0.024), (0.005, -0.05, 0.0)), Part(B, (0.003, 0.053, 0.024), (0.058, 0, 0.0))),
            pivot_z=0.024, kind="slide", axis=(-1, 0, 0), range=(0.0, 0.08), damping=0.5, frictionloss=0.05,
            base_rgba=(0.55, 0.42, 0.28, 1)),
    ),
    ObjectDef(
        "valve", "Valve", "Articulated", "Three-spoke valve on a post (classic dexterous valve-turning).",
        (0.05, 0.05, 0.008), 0.05, (0.85, 0.20, 0.20, 1),
        (Part("cylinder", (0.012, 0.008)),
         Part("capsule", (0.006, 0.038), (0.038, 0, 0), axis="x", name="spoke"),
         Part("capsule", (0.006, 0.038), (-0.019, 0.0329, 0), axis="x", name="spoke_2", yaw_deg=120),
         Part("capsule", (0.006, 0.038), (-0.019, -0.0329, 0), axis="x", name="spoke_3", yaw_deg=240)),
        (T("turn_joint", angle_deg=90),),
        120, "Push a spoke with the fingertips; regrasp to keep turning.", Physics(METAL),
        articulation=Articulation(base_parts=(Part("cylinder", (0.01, 0.03), (0, 0, -0.038)),), pivot_z=0.068),
    ),
    ObjectDef(
        "dial", "Dial knob", "Articulated", "5 cm knob with a pointer on a short base.",
        (0.025, 0.025, 0.015), 0.04, (0.20, 0.22, 0.26, 1),
        (Part("cylinder", (0.025, 0.015)), Part(B, (0.018, 0.003, 0.002), (0.008, 0, 0.0155), name="pointer", rgba=(1, 0.8, 0.1, 1))),
        (T("turn_joint", angle_deg=90),),
        0, "Pinch the rim with thumb and two fingers and twist.", Physics(PLASTIC),
        articulation=Articulation(base_parts=(Part(B, (0.035, 0.035, 0.01), (0, 0, -0.025)),), pivot_z=0.035, damping=0.01),
    ),
    ObjectDef(
        "jar_lid", "Jar lid", "Articulated", "Lid on a fixed glass jar: unscrew it.",
        (0.035, 0.035, 0.012), 0.03, (0.85, 0.70, 0.20, 1),
        (Part("cylinder", (0.035, 0.012)), Part(B, (0.02, 0.003, 0.001), (0.01, 0, 0.0125), name="mark", rgba=(0.2, 0.2, 0.2, 1))),
        (T("turn_joint", angle_deg=180),),
        0, "Spread the fingers around the lid rim and twist counter-clockwise.",
        Physics(METAL, contents="jam (in the jar)", sealed=True),
        articulation=Articulation(base_parts=(Part("cylinder", (0.033, 0.04), (0, 0, -0.052)),), pivot_z=0.092, damping=0.03, frictionloss=0.02,
                                  base_rgba=(0.75, 0.85, 0.95, 0.6)),
    ),
]

OBJECTS_BY_ID = {o.id: o for o in OBJECTS}


def default_part(obj: ObjectDef) -> str | None:
    """The part a 'pick by part' task refers to (first named part)."""
    names = obj.part_names
    return names[0] if names else None
