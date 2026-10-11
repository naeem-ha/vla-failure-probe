"""Tile selected frames of a run video into one PNG. Usage: python frames.py <video.mp4> <out.png> f1 f2 ..."""
import sys

import imageio
import numpy as np

video, out, idx = sys.argv[1], sys.argv[2], [int(i) for i in sys.argv[3:]]
reader = imageio.get_reader(video)
frames = [reader.get_data(i) for i in idx]
imageio.imwrite(out, np.concatenate(frames, axis=1))
