#!/bin/bash
# Static ffmpeg without sudo, via the imageio-ffmpeg wheel.
~/.local/bin/uv pip install --quiet --target ~/.local/share/imageio-ffmpeg imageio-ffmpeg --python python3
bin=$(ls ~/.local/share/imageio-ffmpeg/imageio_ffmpeg/binaries/ffmpeg-* | head -1)
ln -sfn "$bin" ~/.local/bin/ffmpeg
~/.local/bin/ffmpeg -version | head -1
