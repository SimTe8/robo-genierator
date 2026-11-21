"""Download the robot models this repo needs (they are never committed).

    python scripts/setup_assets.py              # menagerie (Panda, Allegro, ...) + Wuji hand
    python scripts/setup_assets.py --only wuji  # just the Wuji hand
    python scripts/setup_assets.py --check      # also compile each robot in MuJoCo

Standard library only, so it runs before any `pip install`. Works on macOS, Linux and Windows.
Each source is pinned to a commit, so every teammate gets byte-identical models. Uses git when it is
installed (sparse checkout: only the menagerie folders we use) and falls back to a GitHub zip download.
Re-running is safe: complete folders are skipped; use --force to re-download.
"""

from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / "assets"


@dataclass(frozen=True)
class Source:
    key: str
    repo: str  # github "owner/name"
    commit: str
    dest: str  # folder under assets/
    license: str
    subdirs: tuple[str, ...] = ()  # sparse: only these top-level folders (empty = whole repo)
    required: tuple[str, ...] = ()  # files that must exist afterwards (relative to dest)


SOURCES = (
    Source(
        key="menagerie",
        repo="google-deepmind/mujoco_menagerie",
        commit="370c43965323c0944b433b202aac5058b5f907a9",
        dest="mujoco_menagerie",
        license="per-model (Apache-2.0 / BSD), see each folder's LICENSE",
        subdirs=("franka_emika_panda", "wonik_allegro", "robotiq_2f85", "universal_robots_ur5e"),
        required=("franka_emika_panda/scene.xml", "franka_emika_panda/panda.xml",
                  "wonik_allegro/scene_right.xml", "wonik_allegro/right_hand.xml"),
    ),
    Source(
        key="wuji",
        repo="wuji-technology/wuji-hand-description",
        commit="bbc2f64eec834aae616eacc2db1f69016f860c2d",
        dest="wuji-hand-description",
        license="MIT (Copyright (c) 2025 Wuji Technology)",
        required=("mjcf/right.xml", "mjcf/left.xml", "LICENSE"),
    ),
)


def log(msg: str) -> None:
    print(msg, flush=True)


def is_complete(src: Source, dest: Path) -> bool:
    return dest.exists() and all((dest / f).is_file() for f in src.required)


def fetch_git(src: Source, dest: Path) -> None:
    url = f"https://github.com/{src.repo}.git"
    tmp = dest.with_name(dest.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(tmp), *args], check=True, stdout=subprocess.DEVNULL)

    git("init", "-q")
    git("remote", "add", "origin", url)
    if src.subdirs:
        git("sparse-checkout", "set", "--no-cone", *[f"/{d}/" for d in src.subdirs], "/LICENSE")
    git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", src.commit)
    git("-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD")
    shutil.rmtree(tmp / ".git", ignore_errors=True)  # plain files: no nested repo inside ours
    tmp.rename(dest)


def fetch_zip(src: Source, dest: Path) -> None:
    url = f"https://codeload.github.com/{src.repo}/zip/{src.commit}"
    log(f"  downloading {url} (no git found)")
    with urllib.request.urlopen(url, timeout=600) as resp:
        data = resp.read()
    tmp = dest.with_name(dest.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            parts = Path(info.filename).parts[1:]  # strip "<repo>-<commit>/"
            if not parts or info.is_dir():
                continue
            if src.subdirs and parts[0] not in src.subdirs and parts != ("LICENSE",):
                continue
            out = tmp.joinpath(*parts)
            if not out.resolve().is_relative_to(tmp.resolve()):
                raise RuntimeError(f"unsafe path in zip: {info.filename}")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(zf.read(info))
    tmp.rename(dest)


def install(src: Source, force: bool) -> None:
    dest = ASSETS / src.dest
    if is_complete(src, dest) and not force:
        log(f"[ok]   {src.dest} already present")
        return
    if dest.is_symlink():
        dest.unlink()
    elif dest.exists():
        if not force:
            raise SystemExit(f"[err]  {dest} exists but is incomplete; re-run with --force to replace it")
        shutil.rmtree(dest)
    log(f"[get]  {src.repo} @ {src.commit[:7]} -> assets/{src.dest}" + (f" (only {', '.join(src.subdirs)})" if src.subdirs else ""))
    ASSETS.mkdir(exist_ok=True)
    if shutil.which("git"):
        fetch_git(src, dest)
    else:
        fetch_zip(src, dest)
    missing = [f for f in src.required if not (dest / f).is_file()]
    if missing:
        raise SystemExit(f"[err]  download finished but files are missing: {missing}")
    log(f"[done] {src.dest}  (license: {src.license})")


def check() -> None:
    """Compile each robot scene in MuJoCo (needs the Python env with mujoco installed)."""
    try:
        import mujoco
    except ImportError:
        log("[skip] --check needs mujoco installed (pip install -r backend/requirements.txt)")
        return
    for rel in ("mujoco_menagerie/franka_emika_panda/scene.xml", "mujoco_menagerie/wonik_allegro/scene_right.xml",
                "wuji-hand-description/mjcf/right.xml"):
        m = mujoco.MjModel.from_xml_path(str(ASSETS / rel))
        log(f"[ok]   MuJoCo {mujoco.__version__} compiled {rel}: {m.nbody} bodies, {m.njnt} joints, {m.nu} actuators")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=[s.key for s in SOURCES], action="append", help="download only this source")
    ap.add_argument("--force", action="store_true", help="re-download even if the folder looks complete")
    ap.add_argument("--check", action="store_true", help="compile the robot models in MuJoCo afterwards")
    args = ap.parse_args()
    for src in SOURCES:
        if not args.only or src.key in args.only:
            install(src, args.force)
    if args.check:
        check()
    log("Robot models ready. They are gitignored: never commit assets/mujoco_menagerie or assets/wuji-hand-description.")


if __name__ == "__main__":
    sys.exit(main())
