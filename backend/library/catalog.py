"""Serialise the object/task library for the UI and turn a selection into planner-ready text."""

from __future__ import annotations

from config import load_config
from library.objects import OBJECTS, OBJECTS_BY_ID, ObjectDef, default_part
from dataclasses import asdict

from library.generated import refresh_generated
from library.scene import resolve_params
from library.tasks import TASKS, TASKS_BY_ID, TaskPreset
from schemas import Catalog, ObjectInfo, ObjectTaskInfo, RobotInfo, TaskParamInfo, TaskPresetInfo

ROBOTS = [
    RobotInfo(id="wonik_allegro", name="Allegro Hand", kind="4-finger hand",
              description="16-DoF hand (index, middle, ring, thumb) on a free-floating mocap base.",
              fingers=["index", "middle", "ring", "thumb"], license="BSD 2-Clause · mujoco_menagerie/wonik_allegro"),
    RobotInfo(id="wuji_hand", name="Wuji Hand", kind="5-finger hand",
              description="20-DoF five-finger hand (4 joints per finger) on a free-floating mocap base.",
              fingers=["thumb", "index", "middle", "ring", "little"], license="MIT · wuji-technology/wuji-hand-description"),
    RobotInfo(id="franka_emika_panda", name="Franka Panda", kind="7-DoF arm + gripper",
              description="7-DoF arm with a parallel-jaw gripper (8 cm max opening). No in-hand manipulation.",
              fingers=["left finger", "right finger"], license="Apache 2.0 · mujoco_menagerie/franka_emika_panda"),
]
ROBOTS_BY_ID = {r.id: r for r in ROBOTS}
BUTTON_NAMES = {1: "red", 2: "green", 3: "blue"}


def _fill(text: str, obj: ObjectDef | None = None, params: dict[str, float] | None = None) -> str:
    values: dict[str, object] = {k: float(v) for k, v in load_config()["checks"].items()}
    values.update(params or {})
    if "button" in values:
        values["button_name"] = BUTTON_NAMES.get(int(values["button"]), "green")  # type: ignore[arg-type]
    if obj is not None:
        values["object"] = obj.name.lower()
        values["part"] = (default_part(obj) or "body").replace("_", " ")
    else:
        values.setdefault("object", "object")
        values.setdefault("part", "part")
    # Whole numbers without ".0" where no format spec is given; unknown placeholders are left as-is.
    pretty = {k: (int(v) if isinstance(v, float) and v.is_integer() and "{" + k + ":" not in text else v) for k, v in values.items()}
    return text.format_map(_Keep(pretty))


class _Keep(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def task_info(task: TaskPreset, obj: ObjectDef | None = None, params: dict[str, float] | None = None) -> TaskPresetInfo:
    return TaskPresetInfo(
        id=task.id,
        name=_fill(task.name, obj, params),
        skill=task.skill,
        summary=_fill(task.summary, obj, params),
        goal=task.goal,
        fixture=task.fixture,
        initial_pose=task.initial_pose,
        params=[TaskParamInfo(**vars(p)) for p in task.params],
        prompt_template=task.prompt,
        criteria=[_fill(c, obj, params) for c in task.criteria],
        robots=list(task.robots),
    )


def object_info(obj: ObjectDef) -> ObjectInfo:
    return ObjectInfo(
        id=obj.id,
        name=obj.name,
        category=obj.category,
        description=obj.description,
        dims_cm=obj.dims_cm,
        mass_kg=obj.mass_kg,
        grasp_note=obj.grasp_note,
        parts=obj.part_names,
        articulated=obj.articulation is not None,
        thumbnail_url=f"/api/objects/{obj.id}/thumb.png",
        physics=asdict(obj.physics),
        tasks=[
            ObjectTaskInfo(
                task_id=r.task_id,
                name=_fill(TASKS_BY_ID[r.task_id].name, obj),
                params=resolve_params(TASKS_BY_ID[r.task_id], obj, None),
            )
            for r in obj.tasks
        ],
    )


def get_catalog() -> Catalog:
    refresh_generated()
    return Catalog(robots=ROBOTS, objects=[object_info(o) for o in OBJECTS], tasks=[task_info(t, None, {p.key: p.default for p in t.params}) for t in TASKS])


def render_prompt(object_id: str, task_id: str, params: dict[str, float]) -> str:
    obj, task = OBJECTS_BY_ID[object_id], TASKS_BY_ID[task_id]
    return _fill(task.prompt, obj, resolve_params(task, obj, params))
