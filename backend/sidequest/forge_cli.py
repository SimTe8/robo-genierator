"""Generate an object from a text prompt with Gemini (side quest).

    python backend/sidequest/forge_cli.py "a TV remote with three buttons"          # draft only
    python backend/sidequest/forge_cli.py "a TV remote with three buttons" --save   # add to the library

Needs GEMINI_API_KEY in the repo-root .env (never in .env.example).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gl_setup  # noqa: E402,F401

from sidequest.asset_forge import forge, save_draft  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("--save", action="store_true", help="add the object to the library")
    args = ap.parse_args()
    draft = forge(args.prompt)
    a = draft["asset"]
    print(f"{a['name']}  ({draft['object_id']}, {len(a['parts'])} parts, {len(a['buttons'])} buttons, {a['mass_kg']} kg)")
    print(f"  tasks: {a['tasks']}")
    print(f"  physics: {json.dumps(a['physics'])}")
    print(f"  warnings: {draft['warnings'] or 'none'}   attempts: {draft['attempts']}")
    print(f"  thumbnail: outputs/drafts/{draft['draft_id']}/thumb.png")
    if args.save:
        print("  saved to library as", save_draft(draft["draft_id"]))


if __name__ == "__main__":
    main()
