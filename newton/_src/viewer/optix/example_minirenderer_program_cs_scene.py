"""Program.cs-style scene replica for the Python MiniRenderer.

Recreates the C# scene pattern:
- load one OBJ mesh
- spawn a 3D grid of instances
- animate per-instance Y rotations each frame
- add a large ground plane
"""

from __future__ import annotations

import argparse
import colorsys
import ctypes
from pathlib import Path

import numpy as np

import warp as wp
from newton._src.viewer.optix.mini_renderer import MiniRenderer, pack_rgba8, rotation_y_mat4


def _animate_instances(renderer: MiniRenderer, instance_handles: list, step: float):
    for i, handle in enumerate(instance_handles):
        sign = 1.0 if (i % 2) == 0 else -1.0
        m = renderer.get_instance_transform_matrix(handle)
        m = m @ rotation_y_mat4(step * sign)
        renderer.set_instance_transform_matrix(handle, m)


def _indexed_color(index: int) -> int:
    # Approximate C# ColorGenerator vibrant palette.
    h = (index * 0.61803398875) % 1.0
    s = 0.55 + 0.35 * (((index * 17) % 11) / 10.0)
    v = 0.72 + 0.24 * (((index * 29) % 13) / 12.0)
    r, g, b = colorsys.hsv_to_rgb(h, min(max(s, 0.0), 1.0), min(max(v, 0.0), 1.0))
    return pack_rgba8(int(255.0 * r), int(255.0 * g), int(255.0 * b), 255)


def _run_gui(renderer: MiniRenderer, fps: int, max_frames: int, animated_instances: list, rotate_step: float):
    import pyglet  # noqa: PLC0415
    from pyglet import gl  # noqa: PLC0415
    from pyglet.window import key, mouse  # noqa: PLC0415

    width = renderer.width
    height = renderer.height
    window = pyglet.window.Window(width=width, height=height, caption="MiniRenderer Program.cs Scene", vsync=False)
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
            # Pan in camera local right/up.
            renderer.camera.move_local(forward=0.0, right=-0.01 * float(dx), up=0.01 * float(dy))

    @window.event
    def on_mouse_scroll(_x, _y, _sx, sy):
        renderer.camera.dolly(amount=0.8 * float(sy))

    @window.event
    def on_key_press(symbol, _modifiers):
        if symbol == key.R:
            renderer.set_camera_pose([35.0, 25.0, 35.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        elif symbol == key.MINUS:
            renderer.camera.fov_y_degrees = float(np.clip(renderer.camera.fov_y_degrees + 2.0, 20.0, 100.0))
        elif symbol == key.EQUAL:
            renderer.camera.fov_y_degrees = float(np.clip(renderer.camera.fov_y_degrees - 2.0, 20.0, 100.0))

    def update(_dt):
        move = 0.35
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

        _animate_instances(renderer, animated_instances, rotate_step)
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
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--frames", type=int, default=0, help="0 means run until window close")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--obj", type=str, default=r"C:\Documents\Meshes\lionFish.obj")
    parser.add_argument("--obj-scale", type=float, default=1.0)
    parser.add_argument("--grid-min", type=int, default=-3)
    parser.add_argument("--grid-max", type=int, default=3)
    parser.add_argument("--grid-spacing", type=float, default=2.5)
    parser.add_argument("--ground-size", type=float, default=100.0)
    parser.add_argument("--ground-y", type=float, default=-10.0)
    parser.add_argument("--rotate-step", type=float, default=0.01)
    args = parser.parse_args()

    renderer = MiniRenderer(width=args.width, height=args.height)

    obj_path = Path(args.obj)
    if obj_path.is_file():
        model = renderer.register_obj(obj_path, scale=args.obj_scale)
    else:
        print(f"Warning: OBJ not found at '{obj_path}', falling back to cube.")
        model = renderer.register_cube(0.8)

    plane = renderer.register_plane(args.ground_size, args.ground_size)

    animated_instances = []
    color_index = 0
    for x in range(args.grid_min, args.grid_max + 1):
        for y in range(args.grid_min, args.grid_max + 1):
            for z in range(args.grid_min, args.grid_max + 1):
                h = renderer.add_render_instance(
                    model,
                    [x * args.grid_spacing, y * args.grid_spacing, z * args.grid_spacing, 0.0, 0.0, 0.0, 1.0],
                    color_rgba8=_indexed_color(color_index),
                )
                animated_instances.append(h)
                color_index += 1

    renderer.add_render_instance(
        plane,
        [0.0, args.ground_y, 0.0, 0.0, 0.0, 0.0, 1.0],
        color_rgba8=pack_rgba8(255, 255, 255, 255),
    )

    renderer.set_camera_pose([35.0, 25.0, 35.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    print("Camera controls: LMB orbit, RMB pan, wheel dolly, WASDQE move, Shift fast, +/- FOV, R reset")
    renderer.rebuild_instance_tree()

    if args.headless:
        frames = args.frames if args.frames > 0 else 60
        for _ in range(frames):
            _animate_instances(renderer, animated_instances, args.rotate_step)
            renderer.render_frame()
        img = renderer.color.numpy().reshape(args.height, args.width)
        checksum = int(np.bitwise_xor.reduce(img.reshape(-1).astype(np.uint32)))
        print(f"Rendered {frames} frames. Checksum: {checksum}")
    else:
        _run_gui(
            renderer,
            fps=args.fps,
            max_frames=args.frames,
            animated_instances=animated_instances,
            rotate_step=args.rotate_step,
        )


if __name__ == "__main__":
    main()
