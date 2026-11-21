"""Pick the right offscreen OpenGL backend. Import this BEFORE anything that imports mujoco.

Linux/WSL2 -> EGL (headless GPU). macOS has no EGL -> CGL. An explicit MUJOCO_GL always wins.
"""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
