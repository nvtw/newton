"""Render a Colibri example screenshot using the Newton GL viewer."""
import argparse
from pathlib import Path

import warp as wp
from newton.examples.kamino.example_kamino_colibri import Example
from newton.viewer import ViewerGL

parser = argparse.ArgumentParser()
parser.add_argument('--body-count', type=int, default=36)
parser.add_argument('--size', type=int, default=640)
parser.add_argument('--frames', type=int, default=0)
parser.add_argument('--output', default='/tmp/colibri.png')
args = parser.parse_args()
wp.init()
viewer = ViewerGL(width=args.size, height=args.size, headless=True)
example = Example(viewer, args)
for _ in range(args.frames):
    example.step()
example.render()
from PIL import Image
Image.fromarray(viewer.get_frame().numpy()).save(args.output)
print(Path(args.output).resolve())
viewer.close()
