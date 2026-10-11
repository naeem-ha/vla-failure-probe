"""Smoke test: build one LIBERO scene, render a frame, save it."""
import pathlib

import imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(7)
env.reset()
obs = env.set_init_state(suite.get_task_init_states(0)[0])
pathlib.Path("runs").mkdir(exist_ok=True)
imageio.imwrite("runs/render_test.png", obs["agentview_image"][::-1, ::-1])
print("TASK:", task.language)
print("RENDER_OK")
