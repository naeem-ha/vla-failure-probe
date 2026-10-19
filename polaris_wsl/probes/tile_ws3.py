"""Tile frames of a video into a grid PNG at reduced size. Usage: python tile_ws3.py <mp4> <out.png> [cols] f1 f2 ..."""
import sys
import imageio
import numpy as np
from PIL import Image
video, out, cols, idx = sys.argv[1], sys.argv[2], int(sys.argv[3]), [int(i) for i in sys.argv[4:]]
r = imageio.get_reader(video)
ims = [Image.fromarray(r.get_data(i)) for i in idx]
w, h = ims[0].width // 2, ims[0].height // 2
rows = (len(ims) + cols - 1) // cols
grid = Image.new("RGB", (w * cols, h * rows))
for k, im in enumerate(ims):
    grid.paste(im.resize((w, h)), ((k % cols) * w, (k // cols) * h))
grid.save(out)
