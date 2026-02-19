"""Python sample inspired by MiniRenderer/PythonSample.py.

API flow:
- register mesh primitives
- add render instances with pose
- set camera pose
- rebuild and render headless
"""

from __future__ import annotations

import argparse
import ctypes
import math

import numpy as np

import warp as wp
from newton._src.viewer.optix.mini_renderer import MiniRenderer


def _run_gui(renderer: MiniRenderer, fps: int, max_frames: int, animated_instance=None):
    import pyglet  # noqa: PLC0415
    from pyglet import gl  # noqa: PLC0415
    from pyglet.window import key, mouse  # noqa: PLC0415

    width = renderer.width
    height = renderer.height

    window = pyglet.window.Window(width=width, height=height, caption="MiniRenderer (Python + OptiX)", vsync=False)
    texture = pyglet.image.Texture.create(width=width, height=height, rectangle=False)
    texture.min_filter = gl.GL_NEAREST
    texture.mag_filter = gl.GL_NEAREST
    sprite = pyglet.sprite.Sprite(texture, x=0, y=0)

    pbo = gl.GLuint()
    gl.glGenBuffers(1, pbo)
    gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo)
    gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, width * height * 4, None, gl.GL_DYNAMIC_DRAW)
    gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

    cuda_gl = wp.RegisteredGLBuffer(
        int(pbo.value), device=renderer.device, flags=wp.RegisteredGLBuffer.WRITE_DISCARD, fallback_to_copy=False
    )

    state = {"frames": 0}
    keys = key.KeyStateHandler()
    window.push_handlers(keys)

    @window.event
    def on_draw():
        window.clear()
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo)
        gl.glBindTexture(texture.target, texture.id)
        gl.glTexSubImage2D(
            texture.target,
            0,
            0,
            0,
            width,
            height,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            ctypes.c_void_p(0),
        )
        gl.glBindTexture(texture.target, 0)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        sprite.draw()

    @window.event
    def on_close():
        gl.glDeleteBuffers(1, pbo)
        window.close()

    @window.event
    def on_deactivate():
        # Avoid pyglet Win32 assert on missing deactivate handler.
        return None

    @window.event
    def on_mouse_drag(_x, _y, dx, dy, buttons, _modifiers):
        if buttons & mouse.LEFT:
            renderer.camera.orbit_target(yaw_radians=-0.003 * float(dx), pitch_radians=0.003 * float(dy))
        elif buttons & mouse.RIGHT:
            renderer.camera.move_local(forward=0.0, right=-0.01 * float(dx), up=0.01 * float(dy))

    @window.event
    def on_mouse_scroll(_x, _y, _sx, sy):
        renderer.camera.dolly(amount=0.8 * float(sy))

    @window.event
    def on_key_press(symbol, _modifiers):
        if symbol == key.R:
            renderer.set_camera_pose([5.0, 5.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])

    def update(_dt):
        move = 0.10
        if keys[key.LSHIFT] or keys[key.RSHIFT]:
            move *= 3.0
        if keys[key.W]:
            renderer.camera.move_local(forward=move)
        if keys[key.S]:
            renderer.camera.move_local(forward=-move)
        if keys[key.A]:
            renderer.camera.move_local(right=-move)
        if keys[key.D]:
            renderer.camera.move_local(right=move)
        if keys[key.Q]:
            renderer.camera.move_local(up=-move)
        if keys[key.E]:
            renderer.camera.move_local(up=move)

        t = state["frames"] / float(max(fps, 1))
        if animated_instance is not None:
            renderer.set_instance_transform(
                animated_instance,
                [1.5 * math.cos(0.7 * t), 1.0, 1.5 * math.sin(0.7 * t), 0.0, 0.0, 0.0, 1.0],
            )
        renderer.render_frame()
        mapped = cuda_gl.map(dtype=wp.uint32, shape=(width * height,))
        wp.copy(mapped, renderer.color)
        cuda_gl.unmap()
        state["frames"] += 1
        if max_frames > 0 and state["frames"] >= max_frames:
            pyglet.app.exit()

    pyglet.clock.schedule_interval(update, 1.0 / float(max(fps, 1)))
    pyglet.app.run()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--frames", type=int, default=0, help="GUI auto-exit frame count (0 = run until window close)")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--obj", type=str, default="", help="Optional OBJ file to load as an additional render mesh")
    parser.add_argument("--obj-scale", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true", help="Disable pyglet viewer and render offscreen")
    args = parser.parse_args()

    renderer = MiniRenderer(width=args.width, height=args.height)

    cube = renderer.register_cube(0.4)
    plane = renderer.register_plane(20.0, 20.0)
    loaded_obj = renderer.register_obj(args.obj, scale=args.obj_scale) if args.obj else None

    # [tx, ty, tz, qx, qy, qz, qw]
    cube_inst = renderer.add_render_instance(cube, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    renderer.add_render_instance(plane, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    if loaded_obj is not None:
        renderer.add_render_instance(loaded_obj, [-1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    renderer.set_camera_pose([5.0, 5.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    print("Camera controls: LMB orbit, RMB pan, wheel dolly, WASDQE move, Shift fast, R reset")
    renderer.rebuild_instance_tree()

    # Animate second position for cube then render
    renderer.set_instance_transform(cube_inst, [1.5, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    if args.headless:
        frames = args.frames if args.frames > 0 else 60
        img = renderer.render_headless(frames=frames)
        checksum = int(np.bitwise_xor.reduce(img.reshape(-1).astype(np.uint32)))
        print(f"Rendered {frames} frames. Checksum: {checksum}")
    else:
        _run_gui(renderer, fps=args.fps, max_frames=args.frames, animated_instance=cube_inst)


if __name__ == "__main__":
    main()
