"""Task presets for the Allegro hand. Each object lists the presets that make sense for it (objects.py).

`goal` tells sim/check.py how to score an attempt (simulator ground truth); `fixture` is what the
scene needs (target pad, bin, ghost target, nail board). Thresholds live in config.yaml under tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    unit: str
    min: float
    max: float
    step: float
    default: float


@dataclass(frozen=True)
class TaskPreset:
    id: str
    name: str
    skill: str  # grasp | in-hand | place | non-prehensile | articulated | tool
    summary: str
    goal: str  # scoring rule in sim/check.py
    prompt: str  # {object}, {part} and params are filled in
    criteria: tuple[str, ...]  # {threshold} names from config.yaml are filled in
    fixture: str | None = None  # target_pad | bin | ghost_table | ghost_air | nail_board
    initial_pose: str = "natural"  # natural | lying (stand_upright starts the object on its side)
    params: tuple[Param, ...] = field(default_factory=tuple)
    robots: tuple[str, ...] = ("wonik_allegro", "wuji_hand", "franka_emika_panda")


HANDS = ("wonik_allegro", "wuji_hand")  # multi-finger hands only (in-hand manipulation)


LIFT = Param("lift_cm", "Lift height", "cm", 3, 20, 1, 10)
ANGLE = Param("angle_deg", "Rotation", "°", 30, 180, 15, 90)

TASKS: list[TaskPreset] = [
    TaskPreset(
        "pick_lift", "Pick & lift", "grasp", "Grasp the object off the table and hold it in the air.",
        "lift", "Pick up the {object} and hold it {lift_cm:.0f} cm above the table.",
        ("Object at least {lift_cm} cm (−{lift_tolerance_cm} cm) above its start", "Still held at the end (in contact with the hand)"),
        params=(LIFT,),
    ),
    TaskPreset(
        "precision_pinch", "Precision pinch", "grasp", "Lift using only the thumb and index fingertips.",
        "pinch", "Pick up the {object} using only a fingertip pinch between the thumb and index finger, and lift it {lift_cm:.0f} cm.",
        ("Object at least {lift_cm} cm (−{lift_tolerance_cm} cm) above its start", "Only thumb and index (Panda: the two gripper fingers) touch it at the end"),
        params=(Param("lift_cm", "Lift height", "cm", 3, 15, 1, 6),),
    ),
    TaskPreset(
        "pick_by_part", "Pick by the {part}", "grasp", "Grasp a specific part (handle, neck, …) and lift.",
        "part_lift", "Grasp the {object} by its {part} and lift it {lift_cm:.0f} cm above the table.",
        ("Object at least {lift_cm} cm (−{lift_tolerance_cm} cm) above its start", "The hand is touching the {part} at the end"),
        params=(LIFT,),
    ),
    TaskPreset(
        "rotate_in_hand", "Rotate in hand", "in-hand", "Lift it, then turn it about the vertical axis without putting it down.",
        "rotate_held", "Pick up the {object}, then rotate it {angle_deg:.0f}° counter-clockwise (seen from above) inside the hand without dropping it.",
        ("Still held above the table at the end", "Yaw changed by the target (±{yaw_tolerance_deg}°)"),
        params=(ANGLE,), robots=HANDS,
    ),
    TaskPreset(
        "reorient_to_target", "Reorient to target", "in-hand", "Match the orientation of the green ghost while holding it.",
        "reorient_held", "Pick up the {object} and reorient it in the hand so it matches the green target orientation shown beside the hand.",
        ("Still held above the table at the end", "Orientation within {orientation_tolerance_deg}° of the target"),
        fixture="ghost_air", params=(ANGLE,), robots=HANDS,
    ),
    TaskPreset(
        "flip_over", "Flip over", "in-hand", "Turn the object upside down and set it back on the table.",
        "flip", "Flip the {object} upside down and leave it resting on the table.",
        ("Turned at least {min_flip_deg}° from its start orientation", "Resting on the table, released"),
    ),
    TaskPreset(
        "place_on_pad", "Place on pad", "place", "Carry it to the green pad and let go.",
        "zone", "Pick up the {object}, place it on the green target pad, and release it.",
        ("Centre inside the pad", "Resting on the table and released"), fixture="target_pad",
    ),
    TaskPreset(
        "place_in_bin", "Place in bin", "place", "Drop it into the bin.",
        "inside", "Pick up the {object} and put it into the bin.",
        ("Inside the bin walls, below the rim", "Released"), fixture="bin",
    ),
    TaskPreset(
        "push_to_pad", "Push to pad", "non-prehensile", "Slide it onto the pad without lifting it.",
        "push", "Without picking it up, push the {object} across the table onto the green target pad.",
        ("Centre inside the pad", "Never lifted more than {max_lift_cm} cm"), fixture="target_pad",
    ),
    TaskPreset(
        "push_to_pose", "Push to pose", "non-prehensile", "Push it into the green ghost outline (position and rotation).",
        "pose", "Push the {object} along the table until it lines up with the green outline, rotated {yaw_deg:.0f}° from where it starts.",
        ("Within {xy_tolerance_cm} cm of the outline", "Rotation within {yaw_tolerance_deg}°", "Never lifted more than {max_lift_cm} cm"),
        fixture="ghost_table", params=(Param("yaw_deg", "Target rotation", "°", 0, 180, 15, 90),),
    ),
    TaskPreset(
        "tip_over", "Tip over", "non-prehensile", "Knock it onto its side without lifting it.",
        "tip", "Tip the {object} over so it lies on its side on the table.",
        ("Tilted {min_tilt_deg}–{max_tilt_deg}° from upright", "Still on the table near where it started"),
    ),
    TaskPreset(
        "stand_upright", "Stand upright", "in-hand", "It starts lying down: stand it up.",
        "upright", "The {object} is lying on its side. Stand it upright on the table.",
        ("Upright within {upright_tolerance_deg}°", "Resting on the table and released"), initial_pose="lying",
    ),
    TaskPreset(
        "turn_part_to_hand", "Turn {part} toward hand", "in-hand", "Rotate it so the {part} points back toward the hand's base.",
        "face_part", "Rotate the {object} so that its {part} points back toward the base of the hand (−x).",
        ("{part} within {facing_tolerance_deg}° of the −x direction", "Upright and on the table"), robots=HANDS,
    ),
    TaskPreset(
        "turn_joint", "Turn it", "articulated", "Twist the hinged part by the target angle.",
        "joint", "Turn the {object} {angle_deg:.0f}° counter-clockwise (seen from above).",
        ("Hinge turned at least the target angle (−{angle_tolerance_deg}°)",),
        params=(Param("angle_deg", "Turn", "°", 30, 360, 15, 90),),
    ),
    TaskPreset(
        "hammer_nail", "Hammer the nail", "tool", "Grasp the hammer and drive the nail into the board.",
        "nail", "Pick up the hammer by its handle and hit the nail until it is driven {depth_cm:.1f} cm into the board.",
        ("Nail driven at least the target depth",), fixture="nail_board",
        params=(Param("depth_cm", "Nail depth", "cm", 0.5, 3, 0.5, 1.5),),
    ),
]

TASKS += [
    # ---- Insertion ------------------------------------------------------------
    TaskPreset(
        "insert_key", "Insert key", "insertion", "Stand the flat key up and push its blade into the keyhole.",
        "key_in", "Pick up the key, turn it so the blade points down, and insert it fully into the keyhole of the lock.",
        ("Key tip at least {key_depth_cm} cm inside the keyhole",), fixture="lock",
    ),
    TaskPreset(
        "unlock_with_key", "Insert & turn key", "insertion", "Insert the key, then twist it to unlock (the lock only turns via the key).",
        "key_turn", "Insert the key into the lock and turn it {angle_deg:.0f}° to unlock it.",
        ("Key tip inside the keyhole", "Lock cylinder turned at least the target angle (−{angle_tolerance_deg}°)"), fixture="lock",
        params=(Param("angle_deg", "Turn", "°", 45, 120, 15, 90),),
    ),
    TaskPreset(
        "insert_matching_hole", "Insert in matching hole", "insertion", "Shape sorter with a round and a square hole: pick the right one.",
        "peg_hole", "Insert the {object} into the matching hole of the shape sorter (there is a round hole and a square hole).",
        ("Peg centred in its matching hole (±{hole_tolerance_cm} cm)", "Peg at least {peg_depth_cm} cm deep"), fixture="shape_board",
    ),
    # ---- Tool use ---------------------------------------------------------------
    TaskPreset(
        "press_tool_button", "Hold & press button", "tool", "Pick the tool up and press its button with a finger while holding it.",
        "tool_button", "Pick up the {object} and, while holding it in the air, press its button with your thumb or index finger.",
        ("Tool held above the table", "Button pressed at least {button_press_fraction:.0%} of its travel"),
    ),
    TaskPreset(
        "drive_screw", "Drive the screw", "tool", "Hold the screwdriver, put the bit on the screw head and press the button.",
        "screw", "Pick up the {object}, place its bit on the screw head in the board, and press the button to drive the screw.",
        ("Bit tip within {bit_tolerance_cm} cm of the screw head", "Button pressed at least {button_press_fraction:.0%} of its travel"),
        fixture="screw_board",
    ),
    # ---- Articulated ---------------------------------------------------------------
    TaskPreset(
        "press_button", "Press a button", "articulated", "Press one named button on the panel.",
        "button", "Press the {button_name} button on the panel and hold it down.",
        ("{button_name} button pressed at least {button_press_fraction:.0%} of its travel", "The other buttons are not pressed"),
        params=(Param("button", "Button (1 red · 2 green · 3 blue)", "", 1, 3, 1, 2),),
    ),
    TaskPreset(
        "press_two_buttons", "Press two buttons at once", "articulated", "Hold the green and blue buttons down at the same time.",
        "two_buttons", "Press and hold the green and the blue buttons at the same time, using two different fingers.",
        ("Green and blue both pressed at the end", "Red not pressed"), robots=HANDS,
    ),
    TaskPreset(
        "open_drawer", "Open the drawer", "articulated", "Pull the drawer out by its handle.",
        "drawer_open", "Grab the drawer handle and pull the drawer open by {open_cm:.0f} cm.",
        ("Drawer open at least the target distance",), params=(Param("open_cm", "Open by", "cm", 3, 8, 1, 6),),
    ),
    TaskPreset(
        "open_drawer_and_place", "Open drawer, put block in", "articulated", "Two steps: open the drawer, then place the small block inside.",
        "drawer_place", "Open the drawer, then pick up the small blue block next to it and put it inside the drawer.",
        ("Drawer open at least {drawer_min_open_cm} cm", "Block inside the drawer"), fixture="cargo",
    ),
    # ---- Physical reasoning (PG-VLM) -------------------------------------------------
    TaskPreset(
        "pour_beads", "Pour into the bowl", "reasoning", "Pour the cup's contents into the bowl without spilling.",
        "pour", "Pick up the cup of beads and pour the beads into the bowl without spilling them on the table.",
        ("At least {pour_fraction:.0%} of the beads end up in the bowl",), fixture="bowl",
    ),
    TaskPreset(
        "lift_heaviest", "Find & lift the heaviest", "reasoning", "Three identical cans; only one is full. Find it by touch and lift it.",
        "heaviest", "Three cans look identical but only one is full. Find the heaviest one by touching or lifting them, then hold it {lift_cm:.0f} cm above the table.",
        ("The heaviest can is lifted at least {lift_cm} cm (−{lift_tolerance_cm} cm)", "The two light cans are back on the table"),
        params=(Param("lift_cm", "Lift height", "cm", 3, 15, 1, 6),),
    ),
    # ---- Multi-object (extra bodies come from library/multi.py) ---------------------
    TaskPreset(
        "stack_tower", "Stack into a tower", "multi-object", "Several copies on the table: stack them all into one tower.",
        "tower", "There are {count:.0f} {object} pieces on the table. Stack all of them on top of each other into one stable tower, then let go.",
        ("All {count:.0f} pieces form one column (each resting on the one below)", "Tower stands on its own, nothing held"),
        params=(Param("count", "Pieces", "", 2, 4, 1, 3),),
    ),
    TaskPreset(
        "stack_on_block", "Stack on the block", "multi-object", "Place the object on top of the wooden base block.",
        "stack_on", "Pick up the {object} and place it on top of the wooden block, then let go.",
        ("Object rests on the top face of the block", "Released, and the block is not knocked over"),
    ),
    TaskPreset(
        "bin_all", "Put them all in the bin", "multi-object", "Clear several copies off the table into the bin, one by one.",
        "all_inside", "There are {count:.0f} {object} pieces on the table. Put every one of them into the bin, one at a time.",
        ("All {count:.0f} pieces inside the bin, below the rim", "Nothing held at the end"),
        fixture="bin", params=(Param("count", "Pieces", "", 2, 4, 1, 3),),
    ),
    TaskPreset(
        "sort_by_color", "Sort by colour", "multi-object", "Red and blue copies: red ones on the red pad, blue ones on the blue pad.",
        "sort", "There are {count:.0f} {object} pieces, some red and some blue. Put each red one on the red pad and each blue one on the blue pad.",
        ("Every piece rests on the pad of its own colour", "Nothing held at the end"),
        fixture="sort_pads", params=(Param("count", "Pieces", "", 2, 4, 1, 4),),
    ),
    TaskPreset(
        "pick_from_clutter", "Pick it out of the clutter", "multi-object", "The target is packed among other objects: lift it without disturbing them.",
        "clutter_pick", "The {object} is surrounded by {count:.0f} other objects. Pick up only the {object} and hold it {lift_cm:.0f} cm above the table without knocking the others.",
        ("Target at least {lift_cm} cm (−{lift_tolerance_cm} cm) above its start, held", "No other object moved more than {clutter_disturb_cm} cm"),
        params=(Param("count", "Other objects", "", 2, 5, 1, 4), Param("lift_cm", "Lift height", "cm", 3, 15, 1, 8)),
    ),
]

TASKS_BY_ID = {t.id: t for t in TASKS}
