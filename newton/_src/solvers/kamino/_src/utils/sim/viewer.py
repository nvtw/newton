# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The customized debug viewer of Kamino"""

# Python
import copy
import glob
import os
import threading
from dataclasses import dataclass
from typing import ClassVar

# Thirdparty
import numpy as np
import warp as wp

from ......geometry.types import GeoType
from ......viewer import ViewerGL
from ...core.builder import ModelBuilderKamino
from ...core.geometry import GeometryDescriptor
from ...core.types import vec3f
from ...core.world import WorldDescriptor
from ...geometry.contacts import ContactMode
from ...utils import logger as msg
from .simulator import Simulator

###
# Kernels
###


@dataclass
class _RenderGroup:
    name: str
    mesh_path: str
    body_indices: np.ndarray
    local_xforms: np.ndarray
    world_offsets: np.ndarray
    xforms_host: np.ndarray
    init_xforms: wp.array[wp.transform]
    scales: wp.array[wp.vec3]
    colors: wp.array[wp.vec3]
    materials: wp.array[wp.vec4]
    initialized: bool = False


def _quat_multiply(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = q_a
    bx, by, bz, bw = q_b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = q[:3]
    t = np.float32(2.0) * np.cross(qv, v)
    return v + q[3] * t + np.cross(qv, t)


def _write_transform_mat44(dst: np.ndarray, index: int, p: np.ndarray, q: np.ndarray) -> None:
    qx, qy, qz, qw = q
    x2 = np.float32(2.0) * qx * qx
    y2 = np.float32(2.0) * qy * qy
    z2 = np.float32(2.0) * qz * qz
    xy = np.float32(2.0) * qx * qy
    xz = np.float32(2.0) * qx * qz
    yz = np.float32(2.0) * qy * qz
    wx = np.float32(2.0) * qw * qx
    wy = np.float32(2.0) * qw * qy
    wz = np.float32(2.0) * qw * qz
    dst[index, :] = (
        1.0 - y2 - z2,
        xy + wz,
        xz - wy,
        0.0,
        xy - wz,
        1.0 - x2 - z2,
        yz + wx,
        0.0,
        xz + wy,
        yz - wx,
        1.0 - x2 - y2,
        0.0,
        p[0],
        p[1],
        p[2],
        1.0,
    )


@wp.kernel
def compute_contact_box_transforms(
    # Kamino contact data
    position_A: wp.array[wp.vec3],  # Contact position on body A
    position_B: wp.array[wp.vec3],  # Contact position on body B
    frame: wp.array[wp.quatf],  # Contact frames
    mode: wp.array[wp.int32],  # Contact modes
    wid: wp.array[wp.int32],
    num_contacts: int,
    world_spacing: wp.vec3,
    box_size: wp.vec3,  # Box dimensions
    # Output buffers
    transforms: wp.array[wp.transform],
    scales: wp.array[wp.vec3],
    colors: wp.array[wp.vec3],
):
    """
    Compute transforms, scales, and colors for contact frame boxes.
    """
    i = wp.tid()

    # Hide contacts beyond the active count or with INACTIVE mode
    contact_mode = mode[i]
    if i >= num_contacts or contact_mode == wp.int32(ContactMode.INACTIVE):
        scales[i] = wp.vec3(0.0, 0.0, 0.0)
        colors[i] = wp.vec3(0.0, 0.0, 0.0)
        transforms[i] = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        return

    # Contact position - we could also use the midpoint?
    contact_pos = position_B[i]

    # Apply world spacing
    world_id = float(wid[i])
    contact_pos = contact_pos + world_spacing * world_id

    # Contact frame rotation
    q = frame[i]

    # Set transform
    transforms[i] = wp.transform(contact_pos, q)

    # Set scale
    scales[i] = box_size

    # Set color based on contact mode
    if contact_mode == wp.int32(ContactMode.OPENING):
        # White
        colors[i] = wp.vec3(1.0, 1.0, 1.0)
    elif contact_mode == wp.int32(ContactMode.STICKING):
        # Black
        colors[i] = wp.vec3(0.1, 0.1, 0.1)
    elif contact_mode == wp.int32(ContactMode.SLIDING):
        # Blue
        colors[i] = wp.vec3(0.404, 0.647, 0.953)
    else:
        # Unknown mode: Gray
        colors[i] = wp.vec3(0.5, 0.5, 0.5)


@wp.kernel
def compute_contact_force_arrows(
    # Kamino contact data
    position_A: wp.array[wp.vec3],
    position_B: wp.array[wp.vec3],
    frame: wp.array[wp.quatf],  # Contact frames
    reaction: wp.array[wp.vec3],  # Contact forces in respective local contact frame
    mode: wp.array[wp.int32],  # Contact modes
    wid: wp.array[wp.int32],
    num_contacts: int,
    world_spacing: wp.vec3,
    force_scale: float,
    force_threshold: float,  # Minimum force to display
    # Output buffers
    line_starts: wp.array[wp.vec3],
    line_ends: wp.array[wp.vec3],
    line_colors: wp.array[wp.vec3],
    line_widths: wp.array[float],
):
    """
    Compute line segments for visualizing contact forces as arrows.
    """
    i = wp.tid()

    if i >= num_contacts:
        return

    # Skip inactive contacts
    if mode[i] == wp.int32(ContactMode.INACTIVE):
        line_starts[i] = wp.vec3(0.0, 0.0, 0.0)
        line_ends[i] = wp.vec3(0.0, 0.0, 0.0)
        line_widths[i] = 0.0
        return

    # Contact position - we could also use the midpoint?
    contact_pos = position_B[i]

    # Apply world spacing
    world_id = float(wid[i])
    contact_pos = contact_pos + world_spacing * world_id

    # Transform force from contact frame to world frame
    # reaction is in contact frame, need to rotate by frame quaternion
    q = frame[i]
    C = wp.quat_to_matrix(q)
    f_world = C * reaction[i]

    # Compute force magnitude
    f_mag = wp.length(f_world)

    # Only render if force is above threshold
    if f_mag < force_threshold:
        line_starts[i] = wp.vec3(0.0, 0.0, 0.0)
        line_ends[i] = wp.vec3(0.0, 0.0, 0.0)
        line_widths[i] = 0.0
        return

    # linear - Nonlinear scaling # todo make this be an option?
    scaled_length = force_scale * f_mag
    # scaled_length = force_scale * wp.sqrt(f_mag)

    # Force direction
    f_dir = f_world / f_mag

    # Arrow from contact point along force direction
    line_starts[i] = contact_pos
    line_ends[i] = contact_pos + f_dir * scaled_length

    # Magenta color for forces
    line_colors[i] = wp.vec3(1.0, 0.0, 1.0)

    # Line width proportional to force magnitude but clipped, requires modification in viewer_gl to work properly
    # We could also use this to actually visualize something meaningful
    line_widths[i] = wp.clamp(1.0 + 0.1 * f_mag, 1.0, 5.0)


###
# Interfaces
###


class ViewerKamino(ViewerGL):
    """
    A customized debug viewer for Kamino.
    """

    # Define a static set of colors for different bodies
    body_colors: ClassVar[list[wp.array]] = [
        wp.array([wp.vec3(0.9, 0.1, 0.3)], dtype=wp.vec3),  # Crimson Red
        wp.array([wp.vec3(0.1, 0.7, 0.9)], dtype=wp.vec3),  # Cyan Blue
        wp.array([wp.vec3(1.0, 0.5, 0.0)], dtype=wp.vec3),  # Orange
        wp.array([wp.vec3(0.6, 0.2, 0.8)], dtype=wp.vec3),  # Purple
        wp.array([wp.vec3(0.2, 0.8, 0.2)], dtype=wp.vec3),  # Green
        wp.array([wp.vec3(0.8, 0.8, 0.2)], dtype=wp.vec3),  # Yellow
        wp.array([wp.vec3(0.8, 0.2, 0.8)], dtype=wp.vec3),  # Magenta
        wp.array([wp.vec3(0.5, 0.5, 0.5)], dtype=wp.vec3),  # Gray
    ]

    # Define a static world spacing offset for multiple worlds
    world_spacing: ClassVar[vec3f] = vec3f(-2.0, 0.0, 0.0)

    def __init__(
        self,
        builder: ModelBuilderKamino,
        simulator: Simulator,
        width: int = 1920,
        height: int = 1080,
        vsync: bool = False,
        headless: bool = False,
        show_contacts: bool = False,
        record_video: bool = False,
        video_folder: str | None = None,
        skip_img_idx: int = 0,
        async_save: bool = False,
    ):
        """
        Initialize the Kamino viewer.

        Args:
            builder: Model builder.
            simulator: The simulator instance to visualize.
            width: Window width in pixels.
            height: Window height in pixels.
            vsync: Enable vertical sync.
            headless: Run without displaying a window.
            show_contacts: Enable contact point visualization (default: False).
            record_video: Enable frame recording to disk.
            video_folder: Directory to save recorded frames (default: "./frames").
            skip_img_idx: Number of initial frames to skip before recording.
            async_save: Save frames asynchronously in background threads.
        """
        # Initialize the base viewer
        super().__init__(width=width, height=height, vsync=vsync, headless=headless)

        # Cache references to the simulator
        self._simulator = simulator

        # Declare and initialize geometry info cache
        self._worlds: list[WorldDescriptor] = builder.worlds
        self._geometry: list[GeometryDescriptor] = copy.deepcopy(list(builder.all_geoms))
        for geom in self._geometry:
            geom.shape = builder.shapes[geom.uid]

        # Initialize video recording settings
        self._record_video = record_video
        self._video_folder = video_folder or "./frames"
        self._async_save = async_save
        self._skip_img_idx = skip_img_idx
        self._img_idx = 0
        self._frame_buffer = None

        # Contact visualization settings
        self._show_contacts = show_contacts

        self._render_groups = self._build_render_groups()

        if self._record_video:
            os.makedirs(self._video_folder, exist_ok=True)

    def _build_render_groups(self) -> list[_RenderGroup]:
        group_data: dict[str, dict[str, object]] = {}
        for geom in self._geometry:
            if geom.shape is None or geom.shape.type == GeoType.NONE:
                continue

            shape = geom.shape
            mesh_path = self._populate_geometry(
                int(shape.type),
                shape.params,
                0.0,
                bool(shape.is_solid),
                geo_src=shape.data if shape.type in (GeoType.MESH, GeoType.CONVEX_MESH, GeoType.HFIELD) else None,
            )
            data = group_data.setdefault(
                mesh_path,
                {
                    "body_indices": [],
                    "local_xforms": [],
                    "world_offsets": [],
                    "colors": [],
                },
            )
            body_index = geom.body + self._worlds[geom.wid].bodies_idx_offset if geom.body >= 0 else -1
            data["body_indices"].append(body_index)
            data["local_xforms"].append(np.asarray(geom.offset, dtype=np.float32))
            data["world_offsets"].append(np.asarray(self.world_spacing, dtype=np.float32) * np.float32(geom.wid))

            color_index = geom.body % len(self.body_colors)
            data["colors"].append(np.asarray(self.body_colors[color_index].numpy()[0], dtype=np.float32))

        groups: list[_RenderGroup] = []
        for idx, (mesh_path, data) in enumerate(group_data.items()):
            count = len(data["body_indices"])
            groups.append(
                _RenderGroup(
                    name=f"/kamino/geoms/group_{idx}",
                    mesh_path=mesh_path,
                    body_indices=np.asarray(data["body_indices"], dtype=np.int32),
                    local_xforms=np.asarray(data["local_xforms"], dtype=np.float32),
                    world_offsets=np.asarray(data["world_offsets"], dtype=np.float32),
                    xforms_host=np.empty((count, 16), dtype=np.float32),
                    init_xforms=wp.array([wp.transform_identity()] * count, dtype=wp.transform, device=self.device),
                    scales=wp.array([wp.vec3(1.0, 1.0, 1.0)] * count, dtype=wp.vec3, device=self.device),
                    colors=wp.array(data["colors"], dtype=wp.vec3, device=self.device),
                    materials=wp.array([wp.vec4(0.5, 0.0, 0.0, 0.0)] * count, dtype=wp.vec4, device=self.device),
                )
            )
        return groups

    def _update_render_group(self, group: _RenderGroup, body_poses: np.ndarray) -> None:
        for i, body_index in enumerate(group.body_indices):
            local = group.local_xforms[i]
            local_p = local[:3]
            local_q = local[3:]
            if body_index >= 0:
                body = body_poses[body_index]
                body_p = body[:3]
                body_q = body[3:]
                p = body_p + _quat_rotate(body_q, local_p)
                q = _quat_multiply(body_q, local_q)
            else:
                p = local_p
                q = local_q
            _write_transform_mat44(group.xforms_host, i, p + group.world_offsets[i], q)

    def _render_geometry_groups(self, body_poses: np.ndarray) -> None:
        for group in self._render_groups:
            self._update_render_group(group, body_poses)
            if not group.initialized:
                self.log_instances(
                    group.name,
                    group.mesh_path,
                    group.init_xforms,
                    group.scales,
                    group.colors,
                    group.materials,
                )
                group.initialized = True

            instancer = self.objects.get(group.name)
            if instancer is not None:
                instancer.update_from_pinned(group.xforms_host, len(group.body_indices))
                instancer.hidden = False

    def render_frame(self, stop_recording: bool = False):
        # Begin a new frame
        self.begin_frame(self.time)

        # Extract body poses from the kamino simulator
        body_poses = self._simulator.state.q_i.numpy()

        self._render_geometry_groups(body_poses)

        # Render contacts if they exist and visualization is enabled
        if hasattr(self._simulator, "contacts") and self._simulator.contacts is not None:
            self.render_contacts_kamino(self._simulator.contacts)

        # End the new frame
        self.end_frame()

        # Capture frame if recording is enabled and not stopped
        if self._record_video and not stop_recording:
            # todo : think about if we should continue to step the _img_idx even when not recording
            self._capture_frame()

    def render_contacts_kamino(self, contacts):
        """
        Render contact points, frames, and forces for contacts.

        Visualizations include:
        - Small oriented boxes showing contact frame by mode
        - Force arrows showing contact force magnitude and direction
        """
        if not self._show_contacts:
            # Hide all contact visualizations
            if hasattr(self, "_contact_box_mesh_created"):
                self.log_instances("/contact_boxes", "/contact_box_mesh", None, None, None, materials=None, hidden=True)
            self.log_lines("/contact_forces", None, None, None)
            return

        # Get number of active contacts
        num_contacts = contacts.model_active_contacts.numpy()[0]
        max_contacts = contacts.model_max_contacts_host

        if False:  # Debug: Always print contact info
            print(f"[VIEWER] Frame {getattr(self, '_frame', 0)}: num_contacts={num_contacts} (max={max_contacts})")

            # Print all contact slots
            modes = contacts.mode.numpy()[:max_contacts]
            positions = contacts.position_B.numpy()[:max_contacts]
            velocities = contacts.velocity.numpy()[:max_contacts]
            reactions = contacts.reaction.numpy()[:max_contacts]

            for i in range(max_contacts):
                active = "ACTIVE" if i < num_contacts else "STALE"
                print(
                    f"  [{active}] Contact[{i}]: mode={modes[i]} (INACTIVE={ContactMode.INACTIVE}), "
                    f"pos={positions[i]}, vel={velocities[i]}, reaction={reactions[i]}"
                )

            self._frame = getattr(self, "_frame", 0) + 1

        # ======================================================================
        # Render Contact Frame Boxes
        # ======================================================================

        # Allocate buffers for box transforms
        if not hasattr(self, "_contact_box_transforms"):
            self._contact_box_transforms = wp.zeros(max_contacts, dtype=wp.transform, device=self.device)
            self._contact_box_scales = wp.zeros(max_contacts, dtype=wp.vec3, device=self.device)
            self._contact_box_colors = wp.zeros(max_contacts, dtype=wp.vec3, device=self.device)

        # Render boxes as instanced meshes
        if not hasattr(self, "_contact_box_mesh_created"):
            # Unit box mesh
            points, indices_wp = self._create_box_mesh_simple(1.0, 1.0, 1.0)
            self.log_mesh(
                "/contact_box_mesh",
                points,
                indices_wp,
                normals=None,
                hidden=True,
            )
            self._contact_box_mesh_created = True

        # Log instances of the box mesh
        if num_contacts > 0:
            # small scaled unit box to show frame orientation
            box_size = wp.vec3(
                0.025, 0.025, 0.025
            )  # a little bit flat would look better? todo should we have like a viewer config somewhere?

            # Compute box transforms, scales, and colors
            wp.launch(
                kernel=compute_contact_box_transforms,
                dim=max_contacts,
                inputs=[
                    contacts.position_A,
                    contacts.position_B,
                    contacts.frame,
                    contacts.mode,
                    contacts.wid,
                    num_contacts,
                    self.world_spacing,
                    box_size,
                ],
                outputs=[
                    self._contact_box_transforms,
                    self._contact_box_scales,
                    self._contact_box_colors,
                ],
                device=self.device,
            )

            # Always render all max_contacts instances, not just active ones
            # Inactive ones will have zero scale from the kernel
            xforms = self._contact_box_transforms
            scales = self._contact_box_scales
            colors = self._contact_box_colors
            self.log_instances(
                "/contact_boxes",
                "/contact_box_mesh",
                xforms,
                scales,
                colors,
                materials=None,
                hidden=False,
            )
        else:
            # Hide instances when no contacts
            if hasattr(self, "_contact_box_mesh_created"):
                self.log_instances(
                    "/contact_boxes",
                    "/contact_box_mesh",
                    None,
                    None,
                    None,
                    materials=None,
                    hidden=True,
                )

        # ======================================================================
        # Render Contact Force Arrow
        # ======================================================================

        # Allocate buffers for force arrows
        if not hasattr(self, "_contact_force_starts"):
            self._contact_force_starts = wp.zeros(max_contacts, dtype=wp.vec3, device=self.device)
            self._contact_force_ends = wp.zeros(max_contacts, dtype=wp.vec3, device=self.device)
            self._contact_force_colors = wp.zeros(max_contacts, dtype=wp.vec3, device=self.device)
            self._contact_force_widths = wp.zeros(max_contacts, dtype=float, device=self.device)

        # Compute force arrows
        wp.launch(
            kernel=compute_contact_force_arrows,
            dim=max_contacts,
            inputs=[
                contacts.position_A,
                contacts.position_B,
                contacts.frame,
                contacts.reaction,
                contacts.mode,
                contacts.wid,
                num_contacts,
                self.world_spacing,
                0.05,  # force_scale # todo move to a cfg file?
                1e-4,  # force_threshold # todo move to a cfg file?
            ],
            outputs=[
                self._contact_force_starts,
                self._contact_force_ends,
                self._contact_force_colors,
                self._contact_force_widths,
            ],
            device=self.device,
        )

        # Render force arrows as lines
        if num_contacts > 0:
            self.log_lines(
                "/contact_forces",
                self._contact_force_starts[:num_contacts],
                self._contact_force_ends[:num_contacts],
                self._contact_force_colors[:num_contacts],
                width=3.0,  # todo this assumes we fix the viewer_gl line width issue
            )
        else:
            self.log_lines("/contact_forces", None, None, None)

    def set_camera_lookat(self, pos: wp.vec3, target: wp.vec3):
        """
        Set the camera position and orient it to face a specific target.

        Args:
            pos: The camera position.
            target: The point the camera should look at.
        """
        # Calculate the direction vector from camera to target
        dir = wp.normalize(target - pos)

        # Calculate camera angles
        yaw = wp.degrees(wp.atan2(dir[1], dir[0]))
        pitch = wp.degrees(wp.asin(dir[2]))

        # Call basic set camera method
        self.set_camera(pos, pitch, yaw)

    def _create_box_mesh_simple(self, sx, sy, sz):
        """
        # todo where should this function go, is it already implemented somewhere else?
        Helper to create a simple box mesh for contact visualization using warp.
        Returns (vertices, indices) as warp arrays.
        """
        # Create vertex array (8 corners of box)
        verts = wp.array(
            [
                wp.vec3(-0.5 * sx, -0.5 * sy, -0.5 * sz),  # 0
                wp.vec3(0.5 * sx, -0.5 * sy, -0.5 * sz),  # 1
                wp.vec3(0.5 * sx, 0.5 * sy, -0.5 * sz),  # 2
                wp.vec3(-0.5 * sx, 0.5 * sy, -0.5 * sz),  # 3
                wp.vec3(-0.5 * sx, -0.5 * sy, 0.5 * sz),  # 4
                wp.vec3(0.5 * sx, -0.5 * sy, 0.5 * sz),  # 5
                wp.vec3(0.5 * sx, 0.5 * sy, 0.5 * sz),  # 6
                wp.vec3(-0.5 * sx, 0.5 * sy, 0.5 * sz),  # 7
            ],
            dtype=wp.vec3,
            device=self.device,
        )

        # Create index array (12 triangles, flattened)
        indices = wp.array(
            [
                # Bottom face
                0,
                2,
                1,
                0,
                3,
                2,
                # Top face
                4,
                5,
                6,
                4,
                6,
                7,
                # Front face
                0,
                1,
                5,
                0,
                5,
                4,
                # Back face
                3,
                7,
                6,
                3,
                6,
                2,
                # Left face
                0,
                4,
                7,
                0,
                7,
                3,
                # Right face
                1,
                2,
                6,
                1,
                6,
                5,
            ],
            dtype=wp.int32,
            device=self.device,
        )

        return verts, indices

    def _capture_frame(self):
        """
        Capture and save a single frame from the viewer.

        This method retrieves the current rendered frame, converts it to a PIL Image,
        and saves it as a PNG file.
        """
        # Attempt to import PIL, which is required for image saving
        try:
            from PIL import Image
        except ImportError:
            msg.warning("PIL not installed. Frames cannot be saved as images.")
            msg.info("Install with: pip install pillow")
            return False

        # Only capture and save if we've reached the skip threshold
        if self._img_idx >= self._skip_img_idx:
            # Get frame from viewer as GPU array (height, width, 3) uint8
            frame = self.get_frame(target_image=self._frame_buffer)

            # Cache buffer for reuse to minimize allocations
            if self._frame_buffer is None:
                self._frame_buffer = frame

            # Convert to numpy on CPU and PIL
            frame_np = frame.numpy()
            image = Image.fromarray(frame_np, mode="RGB")

            # Generate filename with zero-padded frame number # todo : 05d is currently hardcoded
            filename = os.path.join(self._video_folder, f"{self._img_idx - self._skip_img_idx:05d}.png")

            # Save either asynchronously or synchronously
            if self._async_save:
                # Use non-daemon thread to save in background
                # Each image has its own copy, so thread safety is maintained
                threading.Thread(
                    target=image.save,
                    args=(filename,),
                    daemon=False,  # make sure the thread completes even if main program exits todo can be challenged
                ).start()
            else:
                # Synchronous save - blocks until complete
                image.save(filename)

        self._img_idx += 1

    def generate_video(self, output_filename: str = "recording.mp4", fps: int = 60, keep_frames: bool = True) -> bool:
        """
        Generate MP4 video from recorded png frames using imageio-ffmpeg.

        Args:
            output_filename: Name of output video file (default: "recording.mp4")
            fps: Frames per second for video (default: 60)
            keep_frames: If True, keep png frames after video creation; if False, delete them (default: True)
        """
        # Try to import imageio-ffmpeg (optional dependency)
        try:
            import imageio_ffmpeg as ffmpeg  # noqa: PLC0415
        except ImportError:
            msg.warning("imageio-ffmpeg not installed. Frames saved but video not generated.")
            msg.info("Install with: pip install imageio-ffmpeg")
            return False
        # Try to import PIL (optional dependency for image loading)
        try:
            from PIL import Image
        except ImportError:
            msg.warning("PIL not installed. Frames saved but video not generated.")
            msg.info("Install with: pip install pillow")
            return False
        import numpy as np  # noqa: PLC0415

        # Check if we have frames to process
        if not self._record_video or self._img_idx <= self._skip_img_idx:
            msg.warning("No frames recorded, cannot generate video")
            return False

        # Get sorted list of frame files
        frame_files = sorted(glob.glob(os.path.join(self._video_folder, "*.png")))

        if not frame_files:
            msg.warning(f"No png frames found in {self._video_folder}")
            return False

        msg.info(f"Generating video from {len(frame_files)} frames...")
        try:
            # Use imageio-ffmpeg to write video
            writer = ffmpeg.write_frames(
                output_filename,
                size=(self.renderer._screen_width, self.renderer._screen_height),
                fps=fps,
                codec="libx264",
                macro_block_size=8,
                quality=5,  # set to default quality
            )
            writer.send(None)  # Initialize the writer

            # Read each frame and send each frame from and to disk
            for frame_path in frame_files:
                img = Image.open(frame_path)
                frame_array = np.array(img)
                writer.send(frame_array)

            writer.close()
            msg.info(f"Video generated successfully: {output_filename}")

            if not keep_frames:
                msg.info("Deleting png frames...")
                for frame_path in frame_files:
                    os.remove(frame_path)
                msg.info("Frames deleted")

            return True

        except Exception as e:
            msg.warning(f"Failed to generate video: {e}")
            return False
