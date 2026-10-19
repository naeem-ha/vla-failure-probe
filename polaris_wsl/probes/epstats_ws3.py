"""Quick per-episode motion stats from a probe JSON: max lift / xy displacement per object, min ee distance."""
import json, sys
import numpy as np
for f in sys.argv[1:]:
    d = json.load(open(f)); log = d["log"]; s = d["summary"]
    print(f"== {s['probe']} {s['variant']} ic{s['ic']}: unsafe={s['unsafe']} first={s['first_violation_step']} prog={s['rubric_max_progress']} steps={s['steps']}")
    ee = np.array([r["ee_pos"] for r in log])
    for n in log[0]["objects"]:
        p = np.array([r["objects"][n][:3] for r in log])
        lift = p[:, 2] - p[0, 2]; disp = np.linalg.norm(p[:, :2] - p[0, :2], axis=1); dee = np.linalg.norm(ee - p, axis=1)
        print(f"  {n:11s} maxlift {lift.max():.3f} @ {lift.argmax():3d}  maxdisp {disp.max():.3f}  final disp {disp[-1]:.3f}  min ee dist {dee.min():.3f} @ {dee.argmin()}")
