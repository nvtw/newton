# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.utils
from newton._src.solvers.phoenx.constraints.constraint_container import constraint_container_clear_reset_worlds
from newton._src.solvers.phoenx.constraints.constraint_joint import actuated_double_ball_socket_clear_reset_worlds
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_normal_lambda,
    contact_container_clear_reset_worlds,
)
from newton._src.solvers.phoenx.solver_config import PHOENX_CONTACT_MATCHING

from . import g1_recipe
from .env import advance_seed_counter, collect_ppo_rollout, collect_ppo_rollout_seed_counter
from .ppo import BufferRollout, MirrorMapPPO, TrainerPPO

ACTION_DIM_G1 = 29
OBS_DIM_G1_NANOG1 = 98
OBS_DIM_G1_ISAACLAB_FLAT = 99
OBS_DIM_G1 = OBS_DIM_G1_NANOG1
NANOG1_PHASE_PERIOD = g1_recipe.PHASE_PERIOD


def _mjcf_geom_collides(geom: ET.Element) -> bool:
    contype = int(geom.get("contype", "1"))
    conaffinity = int(geom.get("conaffinity", "1"))
    return contype != 0 or conaffinity != 0


def _g1_mjcf_with_visual_meshes_without_mesh_colliders(mjcf_path: Path) -> str:
    """Return G1 MJCF XML that keeps mesh visuals but not mesh colliders."""

    root = ET.parse(mjcf_path).getroot()
    compiler = root.find("compiler")
    mesh_dir = compiler.get("meshdir", ".") if compiler is not None else "."
    mesh_base = (mjcf_path.parent / mesh_dir).resolve()

    for mesh in root.findall(".//asset/mesh"):
        file_attr = mesh.get("file")
        if file_attr is not None and not Path(file_attr).is_absolute():
            mesh.set("file", str(mesh_base / file_attr))
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)

    for parent in root.iter():
        for child in list(parent):
            if child.tag == "geom" and child.get("type") == "mesh" and _mjcf_geom_collides(child):
                parent.remove(child)

    return ET.tostring(root, encoding="unicode")


_G1_OBS_MIRROR_SRC = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    15,
    16,
    17,
    18,
    19,
    20,
    9,
    10,
    11,
    12,
    13,
    14,
    21,
    22,
    23,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    44,
    45,
    46,
    47,
    48,
    49,
    38,
    39,
    40,
    41,
    42,
    43,
    50,
    51,
    52,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    73,
    74,
    75,
    76,
    77,
    78,
    67,
    68,
    69,
    70,
    71,
    72,
    79,
    80,
    81,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    96,
    97,
)
_G1_OBS_MIRROR_SIGN = (
    -1,
    1,
    -1,
    1,
    -1,
    1,
    1,
    -1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    -1,
)
_G1_ACTION_MIRROR_SRC = (
    6,
    7,
    8,
    9,
    10,
    11,
    0,
    1,
    2,
    3,
    4,
    5,
    12,
    13,
    14,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
)
_G1_ACTION_MIRROR_SIGN = (
    1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
)


def g1_mirror_map_ppo() -> MirrorMapPPO:
    """Return nanoG1's validated left/right G1 PPO mirror map."""

    return MirrorMapPPO(
        obs_src=_G1_OBS_MIRROR_SRC,
        obs_sign=_G1_OBS_MIRROR_SIGN,
        action_src=_G1_ACTION_MIRROR_SRC,
        action_sign=_G1_ACTION_MIRROR_SIGN,
    )


_DEFAULT_JOINT_POS_G1 = (
    -0.10,
    0.0,
    0.0,
    0.30,
    -0.20,
    0.0,
    -0.10,
    0.0,
    0.0,
    0.30,
    -0.20,
    0.0,
    0.0,
    0.0,
    0.0,
    0.20,
    0.20,
    0.0,
    1.28,
    0.0,
    0.0,
    0.0,
    0.20,
    -0.20,
    0.0,
    1.28,
    0.0,
    0.0,
    0.0,
)

_UNITREE_KP_G1 = (
    100.0,
    100.0,
    100.0,
    150.0,
    40.0,
    40.0,
    100.0,
    100.0,
    100.0,
    150.0,
    40.0,
    40.0,
    75.0,
    75.0,
    75.0,
    75.0,
    75.0,
    75.0,
    75.0,
    2.0,
    2.0,
    2.0,
    75.0,
    75.0,
    75.0,
    75.0,
    2.0,
    2.0,
    2.0,
)

_UNITREE_KD_G1 = (
    2.0,
    2.0,
    2.0,
    4.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    4.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    0.2,
    0.2,
    0.2,
    2.0,
    2.0,
    2.0,
    2.0,
    0.2,
    0.2,
    0.2,
)

_NANOG1_DOF_DAMPING_G1 = (
    2.0,
    2.0,
    2.0,
    2.0,
    1.0,
    0.2,
    2.0,
    2.0,
    2.0,
    2.0,
    1.0,
    0.2,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    0.2,
    0.2,
    0.2,
    2.0,
    2.0,
    2.0,
    2.0,
    0.2,
    0.2,
    0.2,
)

_NANOG1_DOF_ARMATURE_G1 = (
    0.01017752004,
    0.025101925,
    0.01017752004,
    0.025101925,
    0.00721945,
    0.00721945,
    0.01017752004,
    0.025101925,
    0.01017752004,
    0.025101925,
    0.00721945,
    0.00721945,
    0.01017752004,
    0.00721945,
    0.00721945,
    0.003609725,
    0.003609725,
    0.003609725,
    0.003609725,
    0.003609725,
    0.00425,
    0.00425,
    0.003609725,
    0.003609725,
    0.003609725,
    0.003609725,
    0.003609725,
    0.00425,
    0.00425,
)

_NANOG1_DOF_FRICTIONLOSS_G1 = tuple(0.1 for _ in range(ACTION_DIM_G1))

_DRIVE_KD_G1 = tuple(_UNITREE_KD_G1[i] + _NANOG1_DOF_DAMPING_G1[i] for i in range(ACTION_DIM_G1))
_NANOG1_ACTUATOR_FORCE_KP_G1 = _UNITREE_KP_G1
_NANOG1_ACTUATOR_FORCE_KD_G1 = (
    2.0,
    2.0,
    2.0,
    4.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    4.0,
    2.0,
    2.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)
_NANOG1_ACTUATOR_FORCE_LO_G1 = (
    -88.0,
    -139.0,
    -88.0,
    -139.0,
    -50.0,
    -50.0,
    -88.0,
    -139.0,
    -88.0,
    -139.0,
    -50.0,
    -50.0,
    -88.0,
    -50.0,
    -50.0,
    -25.0,
    -25.0,
    -25.0,
    -25.0,
    -25.0,
    -5.0,
    -5.0,
    -25.0,
    -25.0,
    -25.0,
    -25.0,
    -25.0,
    -5.0,
    -5.0,
)
_NANOG1_ACTUATOR_FORCE_HI_G1 = (
    88.0,
    139.0,
    88.0,
    139.0,
    50.0,
    50.0,
    88.0,
    139.0,
    88.0,
    139.0,
    50.0,
    50.0,
    88.0,
    50.0,
    50.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    5.0,
    5.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    5.0,
    5.0,
)

_CTRL_LO_G1 = (
    -2.5307,
    -0.5236,
    -2.7576,
    -0.087267,
    -0.87267,
    -0.2618,
    -2.5307,
    -0.5236,
    -2.7576,
    -0.087267,
    -0.87267,
    -0.2618,
    -2.618,
    -0.52,
    -0.52,
    -3.0892,
    -1.5882,
    -2.618,
    -1.0472,
    -1.97222,
    -1.61443,
    -1.61443,
    -3.0892,
    -2.2515,
    -2.618,
    -1.0472,
    -1.97222,
    -1.61443,
    -1.61443,
)

_CTRL_HI_G1 = (
    2.8798,
    2.9671,
    2.7576,
    2.8798,
    0.5236,
    0.2618,
    2.8798,
    2.9671,
    2.7576,
    2.8798,
    0.5236,
    0.2618,
    2.618,
    0.52,
    0.52,
    2.6704,
    2.2515,
    2.618,
    2.0944,
    1.97222,
    1.61443,
    1.61443,
    2.6704,
    1.5882,
    2.618,
    2.0944,
    1.97222,
    1.61443,
    1.61443,
)

_NANOG1_CONTACT_GEOMETRY_MJCF = "mjcf"
_NANOG1_CONTACT_GEOMETRY_FOOT_BOXES = "nanog1_foot_boxes"
_NANOG1_CONTACT_GEOMETRIES = (_NANOG1_CONTACT_GEOMETRY_MJCF, _NANOG1_CONTACT_GEOMETRY_FOOT_BOXES)
_G1_REWARD_MODE_NANOG1_DENSE = 0
_G1_REWARD_MODE_SPARSE_COMMAND = 1
_G1_REWARD_MODE_SPARSE_TARGET = 2
_G1_REWARD_MODE_DENSE_SPARSE_COMMAND = 3
_G1_REWARD_MODE_DENSE_TARGET = 4
_G1_REWARD_MODES = {
    "nanog1_dense": _G1_REWARD_MODE_NANOG1_DENSE,
    "sparse_command": _G1_REWARD_MODE_SPARSE_COMMAND,
    "sparse_target": _G1_REWARD_MODE_SPARSE_TARGET,
    "dense_sparse_command": _G1_REWARD_MODE_DENSE_SPARSE_COMMAND,
    "dense_target": _G1_REWARD_MODE_DENSE_TARGET,
}
_G1_OBSERVATION_MODE_NANOG1 = 0
_G1_OBSERVATION_MODE_ISAACLAB_FLAT = 1
_G1_OBSERVATION_MODES = {
    "nanog1": _G1_OBSERVATION_MODE_NANOG1,
    "isaaclab_flat": _G1_OBSERVATION_MODE_ISAACLAB_FLAT,
}
_NANOG1_FOOT_BOX_LOCAL_POS = (0.04, 0.0, -0.029)
_NANOG1_FOOT_BOX_HALF_EXTENTS = (0.09, 0.03, 0.008)
_NANOG1_FOOT_BOX_MU = 0.6


# Frozen from nanoG1's compiled MuJoCo Playground G1 model
# (web/g1_model_const.h). Newton's downloaded Menagerie XML can drift.
_NANOG1_BODY_MASS_G1 = (
    3.8130000000000002,
    1.3500000000000001,
    1.52,
    1.702,
    1.9319999999999999,
    0.073999999999999996,
    0.60799999999999998,
    1.3500000000000001,
    1.52,
    1.702,
    1.9319999999999999,
    0.073999999999999996,
    0.60799999999999998,
    0.214,
    0.085999999999999993,
    7.8179999999999996,
    0.71799999999999997,
    0.64300000000000002,
    0.73399999999999999,
    0.59999999999999998,
    0.085444999999999993,
    0.48404999999999998,
    0.25457600000000002,
    0.71799999999999997,
    0.64300000000000002,
    0.73399999999999999,
    0.59999999999999998,
    0.085444999999999993,
    0.48404999999999998,
    0.25457600000000002,
)
_NANOG1_BODY_COM_G1 = (
    (0, 0, -0.076050000000000006),
    (0.0027409999999999999, 0.047791, -0.02606),
    (0.029812000000000002, -0.0010449999999999999, -0.087933999999999998),
    (-0.057709000000000003, -0.010980999999999999, -0.15078),
    (0.005457, 0.0039639999999999996, -0.12074),
    (-0.0072690000000000003, 0, 0.011136999999999999),
    (0.026505000000000001, 0, -0.016424999999999999),
    (0.0027409999999999999, -0.047791, -0.02606),
    (0.029812000000000002, 0.0010449999999999999, -0.087933999999999998),
    (-0.057709000000000003, 0.010980999999999999, -0.15078),
    (0.005457, -0.0039639999999999996, -0.12074),
    (-0.0072690000000000003, 0, 0.011136999999999999),
    (0.026505000000000001, 0, -0.016424999999999999),
    (0.0034940000000000001, 0.000233, 0.018034000000000001),
    (0, 2.3e-05, 0),
    (0.0020315799999999998, 0.00033968300000000002, 0.18456800000000001),
    (0, 0.035892, -0.011627999999999999),
    (-0.00022699999999999999, 0.0072700000000000004, -0.063242999999999994),
    (0.010773, -0.0029489999999999998, -0.072009000000000004),
    (0.064956, 0.0044539999999999996, -0.010062),
    (0.017139399999999999, 0.00053759100000000002, 4.8864000000000001e-07),
    (0.0229999, -0.00111685, -0.00111658),
    (0.070824399999999996, 0.000191745, 0.0016174200000000001),
    (0, -0.035892, -0.011627999999999999),
    (-0.00022699999999999999, -0.0072700000000000004, -0.063242999999999994),
    (0.010773, 0.0029489999999999998, -0.072009000000000004),
    (0.064956, -0.0044539999999999996, -0.010062),
    (0.017139399999999999, -0.00053759100000000002, 4.8864000000000001e-07),
    (0.0229999, 0.00111685, -0.00111658),
    (0.070824399999999996, -0.000191745, 0.0016174200000000001),
)
_NANOG1_BODY_INERTIA_DIAG_G1 = (
    (0.010548999999999999, 0.0093089000000000002, 0.0079184000000000008),
    (0.0018151700000000001, 0.00153422, 0.00116212),
    (0.0025498600000000001, 0.0024116900000000002, 0.00148755),
    (0.0077616600000000001, 0.0071757499999999998, 0.0016013900000000001),
    (0.011380400000000001, 0.011277799999999999, 0.00146458),
    (1.8899999999999999e-05, 1.40805e-05, 6.9195000000000004e-06),
    (0.0016721800000000001, 0.0016161000000000001, 0.00021762100000000001),
    (0.0018151700000000001, 0.00153422, 0.00116212),
    (0.0025498600000000001, 0.0024116900000000002, 0.00148755),
    (0.0077616600000000001, 0.0071757499999999998, 0.0016013900000000001),
    (0.011374, 0.011284300000000001, 0.00146452),
    (1.8899999999999999e-05, 1.40805e-05, 6.9195000000000004e-06),
    (0.0016721800000000001, 0.0016161000000000001, 0.00021762100000000001),
    (0.00016353099999999999, 0.00010771400000000001, 0.000102205),
    (8.2449999999999998e-06, 7.0790000000000003e-06, 6.3389999999999998e-06),
    (0.121847, 0.10982500000000001, 0.027373499999999999),
    (0.00046586400000000002, 0.00043284200000000001, 0.00040639400000000003),
    (0.00069131100000000001, 0.00061801100000000002, 0.00038897699999999999),
    (0.0010618699999999999, 0.00103217, 0.00040066100000000002),
    (0.00044303499999999999, 0.00042161199999999997, 0.00025935299999999998),
    (5.4821100000000003e-05, 4.9664599999999999e-05, 3.5779799999999999e-05),
    (0.00043035300000000002, 0.000429873, 0.00016464800000000001),
    (0.00064611300000000003, 0.00055999299999999995, 0.00014756599999999999),
    (0.00046586400000000002, 0.00043284200000000001, 0.00040639400000000003),
    (0.00069131100000000001, 0.00061801100000000002, 0.00038897699999999999),
    (0.0010618699999999999, 0.00103217, 0.00040066100000000002),
    (0.00044303499999999999, 0.00042161199999999997, 0.00025935299999999998),
    (5.4821100000000003e-05, 4.9664599999999999e-05, 3.5779799999999999e-05),
    (0.00043035300000000002, 0.000429873, 0.00016464800000000001),
    (0.00064611300000000003, 0.00055999299999999995, 0.00014756599999999999),
)
_NANOG1_BODY_INERTIA_QUAT_WXYZ_G1 = (
    (0.99999992034044649, 0, -0.00039914796820404854, 0),
    (0.95486247334675789, 0.29396414572462443, 0.03025561499838738, 0.030122014932158825),
    (0.97780777271005237, -1.9711895417999528e-05, 0.20557595221417879, -0.040379290613894565),
    (0.60059791618986991, 0.15831997790731939, 0.22348196881432258, 0.75118089517684494),
    (0.92341787094120198, -0.032769895420011408, 0.015824597788321373, 0.38206694660153068),
    (0.60305298707416244, 0.36922499208603166, 0.36922499208603166, 0.60305298707416244),
    (-0.00048109183327316559, 0.72848174753789752, -0.00061896678549132279, 0.68506476258445614),
    (0.95486247334675789, -0.29396414572462443, 0.03025561499838738, -0.030122014932158825),
    (0.97780777271005237, 1.9711895417999528e-05, 0.20557595221417879, 0.040379290613894565),
    (0.75118089517684494, 0.22348196881432258, 0.15831997790731939, 0.60059791618986991),
    (0.9234388874809869, 0.034527595792887805, 0.011633298582507955, -0.38201195345267719),
    (0.60305298707416244, 0.36922499208603166, 0.36922499208603166, 0.60305298707416244),
    (0.00048109183327316559, 0.72848174753789752, 0.00061896678549132279, 0.68506476258445614),
    (0.28969698640220309, 0.59100097225959691, -0.33779498414457937, 0.67282096841913008),
    (0.5, 0.5, -0.5, 0.5),
    (0.99980258063553606, -6.0331874693959809e-05, 0.019825591684209672, 0.0013198594463885572),
    (0.65415191829949193, 0.013045798370640941, -0.32626695925078625, 0.68224991479018404),
    (0.70125602523352781, -0.019622300706075745, -0.0071031702555957275, 0.71260402564186665),
    (0.71687871474575415, -0.096482861608365042, -0.067994172944340348, 0.68713372658162541),
    (0.54176523639091512, 0.6361322775665198, 0.38882116965612767, 0.38812916935418396),
    (0.57533836856043863, 0.41166726371310447, -0.57490636828370023, 0.41109426334604177),
    (0.24999782840992438, 0.66136254606305978, 0.29303579887011338, 0.64360755824948446),
    (0.51057084925551066, 0.52629484461304887, 0.4680778618014359, 0.49318785438778706),
    (0.68224991479018404, -0.32626695925078625, 0.013045798370640941, 0.65415191829949193),
    (0.71260402564186665, -0.0071031702555957275, -0.019622300706075745, 0.70125602523352781),
    (0.68713372658162541, -0.067994172944340348, -0.096482861608365042, 0.71687871474575415),
    (0.38812916935418396, 0.38882116965612767, 0.6361322775665198, 0.54176523639091512),
    (0.41166726371310447, 0.57533836856043863, -0.41109426334604177, 0.57490636828370023),
    (0.64360755824948446, 0.29303579887011338, 0.66136254606305978, 0.24999782840992438),
    (0.49318785438778706, 0.4680778618014359, 0.52629484461304887, 0.51057084925551066),
)


def _quat_wxyz_to_matrix_np(quat_wxyz: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quat_wxyz
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
            (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
            (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _nanog1_body_inertia_matrix(
    diag: tuple[float, float, float], quat_wxyz: tuple[float, float, float, float]
) -> wp.mat33:
    rot = _quat_wxyz_to_matrix_np(quat_wxyz)
    inertia = rot @ np.diag(np.asarray(diag, dtype=np.float64)) @ rot.T
    return wp.mat33(
        float(inertia[0, 0]),
        float(inertia[0, 1]),
        float(inertia[0, 2]),
        float(inertia[1, 0]),
        float(inertia[1, 1]),
        float(inertia[1, 2]),
        float(inertia[2, 0]),
        float(inertia[2, 1]),
        float(inertia[2, 2]),
    )


def _apply_nanog1_body_inertials(builder: newton.ModelBuilder) -> None:
    if len(builder.body_mass) != len(_NANOG1_BODY_MASS_G1):
        raise RuntimeError(
            f"expected {len(_NANOG1_BODY_MASS_G1)} G1 bodies before nanoG1 inertial override, "
            f"got {len(builder.body_mass)}"
        )
    for body, (mass, com, diag, quat_wxyz) in enumerate(
        zip(
            _NANOG1_BODY_MASS_G1,
            _NANOG1_BODY_COM_G1,
            _NANOG1_BODY_INERTIA_DIAG_G1,
            _NANOG1_BODY_INERTIA_QUAT_WXYZ_G1,
            strict=True,
        )
    ):
        builder.body_mass[body] = float(mass)
        builder.body_inv_mass[body] = 0.0 if mass == 0.0 else 1.0 / float(mass)
        builder.body_com[body] = wp.vec3(float(com[0]), float(com[1]), float(com[2]))
        builder.body_inertia[body] = _nanog1_body_inertia_matrix(diag, quat_wxyz)


@wp.func
def _clip_float(x: wp.float32, lo: wp.float32, hi: wp.float32) -> wp.float32:
    return wp.min(wp.max(x, lo), hi)


@wp.func
def _finite_or_zero(x: wp.float32) -> wp.float32:
    if wp.isfinite(x):
        return x
    return wp.float32(0.0)


@wp.func
def _clip_finite(x: wp.float32, lo: wp.float32, hi: wp.float32) -> wp.float32:
    return _clip_float(_finite_or_zero(x), lo, hi)


def _g1_reward_mode_id(reward_mode: str) -> int:
    try:
        return _G1_REWARD_MODES[str(reward_mode)]
    except KeyError as exc:
        modes = ", ".join(sorted(_G1_REWARD_MODES))
        raise ValueError(f"reward_mode must be one of: {modes}") from exc


def _g1_observation_mode_id(observation_mode: str) -> int:
    try:
        return _G1_OBSERVATION_MODES[str(observation_mode)]
    except KeyError as exc:
        modes = ", ".join(sorted(_G1_OBSERVATION_MODES))
        raise ValueError(f"observation_mode must be one of: {modes}") from exc


def _g1_observation_dim(observation_mode_id: int) -> int:
    if int(observation_mode_id) == _G1_OBSERVATION_MODE_ISAACLAB_FLAT:
        return OBS_DIM_G1_ISAACLAB_FLAT
    return OBS_DIM_G1_NANOG1


@wp.func
def _quat_rotate_inverse_wxyz(qw: wp.float32, qx: wp.float32, qy: wp.float32, qz: wp.float32, v: wp.vec3) -> wp.vec3:
    q = wp.vec3(qx, qy, qz)
    a = v * (wp.float32(2.0) * qw * qw - wp.float32(1.0))
    b = wp.cross(q, v) * qw * wp.float32(2.0)
    c = q * (wp.dot(q, v) * wp.float32(2.0))
    return a - b + c


@wp.func
def _quat_rotate_wxyz(qw: wp.float32, qx: wp.float32, qy: wp.float32, qz: wp.float32, v: wp.vec3) -> wp.vec3:
    q = wp.vec3(qx, qy, qz)
    a = v * (wp.float32(2.0) * qw * qw - wp.float32(1.0))
    b = wp.cross(q, v) * qw * wp.float32(2.0)
    c = q * (wp.dot(q, v) * wp.float32(2.0))
    return a + b + c


@wp.kernel
def g1_apply_actions_kernel(
    actions: wp.array2d[wp.float32],
    default_joint_pos: wp.array[wp.float32],
    ctrl_lower: wp.array[wp.float32],
    ctrl_upper: wp.array[wp.float32],
    action_scale: wp.float32,
    controlled_action_count: wp.int32,
    dof_stride: wp.int32,
    coord_stride: wp.int32,
    target_uses_coord_layout: wp.int32,
    current_actions: wp.array2d[wp.float32],
    joint_target_q: wp.array[wp.float32],
):
    world, action = wp.tid()
    value = _clip_float(actions[world, action], wp.float32(-1.0), wp.float32(1.0))
    if action >= controlled_action_count:
        value = wp.float32(0.0)
    current_actions[world, action] = value

    target = default_joint_pos[action] + action_scale * value
    target = _clip_float(target, ctrl_lower[action], ctrl_upper[action])
    if target_uses_coord_layout != 0:
        joint_target_q[world * coord_stride + wp.int32(7) + action] = target
    else:
        joint_target_q[world * dof_stride + wp.int32(6) + action] = target


@wp.kernel
def g1_observe_reward_kernel(
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    default_joint_pos: wp.array[wp.float32],
    current_actions: wp.array2d[wp.float32],
    previous_actions: wp.array2d[wp.float32],
    command: wp.array2d[wp.float32],
    target_position: wp.array2d[wp.float32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    foot_contacts: wp.array2d[wp.float32],
    actuator_force: wp.array2d[wp.float32],
    coord_stride: wp.int32,
    dof_stride: wp.int32,
    body_stride: wp.int32,
    left_foot_body: wp.int32,
    right_foot_body: wp.int32,
    episode_steps: wp.array[wp.int32],
    max_episode_steps: wp.int32,
    phase_period: wp.int32,
    min_base_height: wp.float32,
    min_upright_cos: wp.float32,
    max_abs_root_position: wp.float32,
    max_abs_root_linear_velocity: wp.float32,
    max_abs_root_angular_velocity: wp.float32,
    max_abs_joint_position: wp.float32,
    max_abs_joint_velocity: wp.float32,
    action_scale: wp.float32,
    reward_dt: wp.float32,
    w_track_lin: wp.float32,
    w_track_ang: wp.float32,
    w_command_progress: wp.float32,
    w_lin_vel_z: wp.float32,
    w_ang_vel_xy: wp.float32,
    w_orientation: wp.float32,
    w_torque: wp.float32,
    w_action_rate: wp.float32,
    w_alive: wp.float32,
    w_termination: wp.float32,
    gait_stance_fraction: wp.float32,
    w_gait_contact: wp.float32,
    w_gait_swing: wp.float32,
    w_gait_swing_contact: wp.float32,
    w_gait_hip: wp.float32,
    gait_foot_height: wp.float32,
    w_base_height: wp.float32,
    base_height_target: wp.float32,
    w_feet_air_time: wp.float32,
    feet_air_time_threshold: wp.float32,
    w_feet_slide: wp.float32,
    ctrl_lower: wp.array[wp.float32],
    ctrl_upper: wp.array[wp.float32],
    w_joint_deviation_hip: wp.float32,
    w_joint_deviation_waist: wp.float32,
    w_joint_deviation_upper: wp.float32,
    w_joint_acc_legs: wp.float32,
    w_joint_pos_limit_ankle: wp.float32,
    reward_mode: wp.int32,
    observation_mode: wp.int32,
    w_sparse_command_success: wp.float32,
    w_target_progress: wp.float32,
    sparse_command_velocity_tolerance: wp.float32,
    sparse_command_yaw_tolerance: wp.float32,
    sparse_target_radius: wp.float32,
    sparse_target_success_upright_cos: wp.float32,
    sparse_target_success_min_base_height: wp.float32,
    sparse_target_success_max_base_height: wp.float32,
    w_mechanical_power: wp.float32,
    foot_air_time: wp.array2d[wp.float32],
    foot_contact_time: wp.array2d[wp.float32],
    foot_timer_episode_step: wp.array[wp.int32],
    previous_joint_qd: wp.array2d[wp.float32],
    joint_acc_l2: wp.array[wp.float32],
    joint_regularizer_episode_step: wp.array[wp.int32],
    obs: wp.array2d[wp.float32],
    rewards: wp.array[wp.float32],
    dones: wp.array[wp.float32],
    successes: wp.array[wp.float32],
):
    world, col = wp.tid()
    q_base = world * coord_stride
    qd_base = world * dof_stride

    qx = joint_q[q_base + wp.int32(3)]
    qy = joint_q[q_base + wp.int32(4)]
    qz = joint_q[q_base + wp.int32(5)]
    qw = joint_q[q_base + wp.int32(6)]
    lin_com_w = wp.vec3(joint_qd[qd_base], joint_qd[qd_base + wp.int32(1)], joint_qd[qd_base + wp.int32(2)])
    ang_w = wp.vec3(
        joint_qd[qd_base + wp.int32(3)],
        joint_qd[qd_base + wp.int32(4)],
        joint_qd[qd_base + wp.int32(5)],
    )
    root_com_b = body_com[world * body_stride]
    root_com_w = _quat_rotate_wxyz(qw, qx, qy, qz, root_com_b)
    lin_origin_w = lin_com_w - wp.cross(ang_w, root_com_w)
    lin_b = _quat_rotate_inverse_wxyz(qw, qx, qy, qz, lin_origin_w)
    ang_b = _quat_rotate_inverse_wxyz(qw, qx, qy, qz, ang_w)
    gravity_b = _quat_rotate_inverse_wxyz(qw, qx, qy, qz, wp.vec3(0.0, 0.0, -1.0))
    target_delta_w = wp.vec3(
        target_position[world, 0] - joint_q[q_base],
        target_position[world, 1] - joint_q[q_base + wp.int32(1)],
        wp.float32(0.0),
    )
    target_delta_b = _quat_rotate_inverse_wxyz(qw, qx, qy, qz, target_delta_w)

    state_bad = wp.int32(0)
    if not wp.isfinite(qw) or not wp.isfinite(qx) or not wp.isfinite(qy) or not wp.isfinite(qz):
        state_bad = wp.int32(1)
    for i in range(3):
        root_q = joint_q[q_base + i]
        if not wp.isfinite(root_q):
            state_bad = wp.int32(1)
        if max_abs_root_position > wp.float32(0.0) and wp.abs(root_q) > max_abs_root_position:
            state_bad = wp.int32(1)
    if max_abs_root_linear_velocity > wp.float32(0.0):
        if wp.dot(lin_origin_w, lin_origin_w) > max_abs_root_linear_velocity * max_abs_root_linear_velocity:
            state_bad = wp.int32(1)
    if max_abs_root_angular_velocity > wp.float32(0.0):
        if wp.dot(ang_w, ang_w) > max_abs_root_angular_velocity * max_abs_root_angular_velocity:
            state_bad = wp.int32(1)
    for j in range(ACTION_DIM_G1):
        joint_pos = joint_q[q_base + wp.int32(7) + j]
        joint_vel = joint_qd[qd_base + wp.int32(6) + j]
        if not wp.isfinite(joint_pos) or not wp.isfinite(joint_vel):
            state_bad = wp.int32(1)
        if max_abs_joint_position > wp.float32(0.0) and wp.abs(joint_pos) > max_abs_joint_position:
            state_bad = wp.int32(1)
        if max_abs_joint_velocity > wp.float32(0.0) and wp.abs(joint_vel) > max_abs_joint_velocity:
            state_bad = wp.int32(1)

    value = wp.float32(0.0)
    if observation_mode == wp.int32(1):
        if col < wp.int32(3):
            value = _clip_finite(lin_b[col], wp.float32(-50.0), wp.float32(50.0))
        elif col < wp.int32(6):
            value = _clip_finite(ang_b[col - wp.int32(3)], wp.float32(-50.0), wp.float32(50.0))
        elif col < wp.int32(9):
            value = _clip_finite(gravity_b[col - wp.int32(6)], wp.float32(-1.0), wp.float32(1.0))
        elif col < wp.int32(12):
            command_col = col - wp.int32(9)
            value = command[world, command_col]
            if (reward_mode == wp.int32(2) or reward_mode == wp.int32(4)) and command_col < wp.int32(2):
                value = target_delta_b[command_col]
        elif col < wp.int32(41):
            j = col - wp.int32(12)
            value = _clip_finite(
                joint_q[q_base + wp.int32(7) + j] - default_joint_pos[j], wp.float32(-10.0), wp.float32(10.0)
            )
        elif col < wp.int32(70):
            j = col - wp.int32(41)
            value = _clip_finite(joint_qd[qd_base + wp.int32(6) + j], wp.float32(-200.0), wp.float32(200.0))
        else:
            value = current_actions[world, col - wp.int32(70)]
    else:
        if col < wp.int32(3):
            value = wp.float32(0.25) * _clip_finite(ang_b[col], wp.float32(-40.0), wp.float32(40.0))
        elif col < wp.int32(6):
            value = _clip_finite(gravity_b[col - wp.int32(3)], wp.float32(-1.0), wp.float32(1.0))
        elif col < wp.int32(9):
            command_col = col - wp.int32(6)
            value = command[world, command_col]
            if (reward_mode == wp.int32(2) or reward_mode == wp.int32(4)) and command_col < wp.int32(2):
                value = target_delta_b[command_col]
        elif col < wp.int32(38):
            j = col - wp.int32(9)
            value = _clip_finite(
                joint_q[q_base + wp.int32(7) + j] - default_joint_pos[j], wp.float32(-10.0), wp.float32(10.0)
            )
        elif col < wp.int32(67):
            j = col - wp.int32(38)
            value = wp.float32(0.05) * _clip_finite(
                joint_qd[qd_base + wp.int32(6) + j], wp.float32(-200.0), wp.float32(200.0)
            )
        elif col < wp.int32(96):
            value = current_actions[world, col - wp.int32(67)]
        elif col == wp.int32(96):
            phase_step = episode_steps[world] % phase_period
            phase = wp.float32(6.283185307179586) * wp.float32(phase_step) / wp.float32(phase_period)
            value = wp.sin(phase)
        else:
            phase_step = episode_steps[world] % phase_period
            phase = wp.float32(6.283185307179586) * wp.float32(phase_step) / wp.float32(phase_period)
            value = wp.cos(phase)
    obs[world, col] = _clip_finite(value, wp.float32(-100.0), wp.float32(100.0))

    if col == 0:
        lin_b_x = _clip_finite(lin_b[0], wp.float32(-50.0), wp.float32(50.0))
        lin_b_y = _clip_finite(lin_b[1], wp.float32(-50.0), wp.float32(50.0))
        lin_b_z = _clip_finite(lin_b[2], wp.float32(-50.0), wp.float32(50.0))
        ang_x = _clip_finite(ang_b[0], wp.float32(-50.0), wp.float32(50.0))
        ang_y = _clip_finite(ang_b[1], wp.float32(-50.0), wp.float32(50.0))
        ang_z = _clip_finite(ang_b[2], wp.float32(-50.0), wp.float32(50.0))
        gravity_x = _clip_finite(gravity_b[0], wp.float32(-1.0), wp.float32(1.0))
        gravity_y = _clip_finite(gravity_b[1], wp.float32(-1.0), wp.float32(1.0))
        gravity_z = _clip_finite(gravity_b[2], wp.float32(-1.0), wp.float32(1.0))
        vx_err = _clip_finite(command[world, 0] - lin_b_x, wp.float32(-20.0), wp.float32(20.0))
        vy_err = _clip_finite(command[world, 1] - lin_b_y, wp.float32(-20.0), wp.float32(20.0))
        yaw_err = _clip_finite(command[world, 2] - ang_z, wp.float32(-20.0), wp.float32(20.0))
        track_lin = wp.exp(-(vx_err * vx_err + vy_err * vy_err) / wp.float32(0.25))
        track_ang = wp.exp(-(yaw_err * yaw_err) / wp.float32(0.25))
        command_progress = command[world, 0] * lin_b_x + command[world, 1] * lin_b_y + command[world, 2] * ang_z
        lin_vel_z_penalty = lin_b_z * lin_b_z
        ang_vel_xy_penalty = ang_x * ang_x + ang_y * ang_y
        orientation_penalty = gravity_x * gravity_x + gravity_y * gravity_y
        upright_cos = _clip_float(-gravity_z, wp.float32(0.0), wp.float32(1.0))
        upright_gate = _clip_float((upright_cos - wp.float32(0.75)) * wp.float32(4.0), wp.float32(0.0), wp.float32(1.0))

        action_rate_penalty = wp.float32(0.0)
        torque_sq_penalty = wp.float32(0.0)
        mechanical_power_penalty = wp.float32(0.0)
        for j in range(ACTION_DIM_G1):
            da = current_actions[world, j] - previous_actions[world, j]
            action_rate_penalty = action_rate_penalty + da * da
            force = _clip_finite(actuator_force[world, j], wp.float32(-10000.0), wp.float32(10000.0))
            torque_sq_penalty = torque_sq_penalty + force * force
            qd_joint = _clip_finite(joint_qd[qd_base + wp.int32(6) + j], wp.float32(-200.0), wp.float32(200.0))
            mechanical_power_penalty = mechanical_power_penalty + wp.abs(force * qd_joint)

        left_contact = wp.int32(0)
        right_contact = wp.int32(0)
        gait_reward = wp.float32(0.0)
        feet_air_time_reward = wp.float32(0.0)
        feet_slide_penalty = wp.float32(0.0)
        if left_foot_body >= wp.int32(0) and right_foot_body >= wp.int32(0):
            if foot_contacts[world, 0] > wp.float32(0.5):
                left_contact = wp.int32(1)
            if foot_contacts[world, 1] > wp.float32(0.5):
                right_contact = wp.int32(1)

            current_step = episode_steps[world]
            if current_step > foot_timer_episode_step[world]:
                foot_timer_episode_step[world] = current_step
                if current_step <= wp.int32(0):
                    foot_air_time[world, 0] = wp.float32(0.0)
                    foot_air_time[world, 1] = wp.float32(0.0)
                    foot_contact_time[world, 0] = wp.float32(0.0)
                    foot_contact_time[world, 1] = wp.float32(0.0)
                else:
                    if left_contact != wp.int32(0):
                        foot_contact_time[world, 0] = foot_contact_time[world, 0] + reward_dt
                        foot_air_time[world, 0] = wp.float32(0.0)
                    else:
                        foot_air_time[world, 0] = foot_air_time[world, 0] + reward_dt
                        foot_contact_time[world, 0] = wp.float32(0.0)
                    if right_contact != wp.int32(0):
                        foot_contact_time[world, 1] = foot_contact_time[world, 1] + reward_dt
                        foot_air_time[world, 1] = wp.float32(0.0)
                    else:
                        foot_air_time[world, 1] = foot_air_time[world, 1] + reward_dt
                        foot_contact_time[world, 1] = wp.float32(0.0)

            command_speed_sq = command[world, 0] * command[world, 0] + command[world, 1] * command[world, 1]
            if command_speed_sq > wp.float32(0.01):
                if left_contact + right_contact == wp.int32(1):
                    left_mode_time = foot_air_time[world, 0]
                    if left_contact != wp.int32(0):
                        left_mode_time = foot_contact_time[world, 0]
                    right_mode_time = foot_air_time[world, 1]
                    if right_contact != wp.int32(0):
                        right_mode_time = foot_contact_time[world, 1]
                    feet_air_time_reward = wp.min(wp.min(left_mode_time, right_mode_time), feet_air_time_threshold)

            if left_contact != wp.int32(0):
                left_vel = wp.spatial_top(body_qd[world * body_stride + left_foot_body])
                feet_slide_penalty = feet_slide_penalty + wp.sqrt(left_vel[0] * left_vel[0] + left_vel[1] * left_vel[1])
            if right_contact != wp.int32(0):
                right_vel = wp.spatial_top(body_qd[world * body_stride + right_foot_body])
                feet_slide_penalty = feet_slide_penalty + wp.sqrt(
                    right_vel[0] * right_vel[0] + right_vel[1] * right_vel[1]
                )

            phase_step = episode_steps[world] % phase_period
            left_phase = wp.float32(phase_step) / wp.float32(phase_period)
            right_phase = left_phase + wp.float32(0.5)
            if right_phase >= wp.float32(1.0):
                right_phase = right_phase - wp.float32(1.0)

            left_stance = wp.int32(0)
            if left_phase < gait_stance_fraction:
                left_stance = wp.int32(1)
            right_stance = wp.int32(0)
            if right_phase < gait_stance_fraction:
                right_stance = wp.int32(1)

            if left_stance == left_contact:
                gait_reward = gait_reward + w_gait_contact
            if right_stance == right_contact:
                gait_reward = gait_reward + w_gait_contact
            if left_stance == wp.int32(0) and left_contact != wp.int32(0):
                gait_reward = gait_reward + w_gait_swing_contact
            if right_stance == wp.int32(0) and right_contact != wp.int32(0):
                gait_reward = gait_reward + w_gait_swing_contact

            if left_contact == wp.int32(0):
                left_foot_z = wp.transform_get_translation(body_q[world * body_stride + left_foot_body])[2]
                left_dz = _clip_finite(left_foot_z - gait_foot_height, wp.float32(-10.0), wp.float32(10.0))
                gait_reward = gait_reward + w_gait_swing * left_dz * left_dz
            if right_contact == wp.int32(0):
                right_foot_z = wp.transform_get_translation(body_q[world * body_stride + right_foot_body])[2]
                right_dz = _clip_finite(right_foot_z - gait_foot_height, wp.float32(-10.0), wp.float32(10.0))
                gait_reward = gait_reward + w_gait_swing * right_dz * right_dz

            hip_penalty = wp.float32(0.0)
            hip_dq = joint_q[q_base + wp.int32(7) + wp.int32(1)] - default_joint_pos[wp.int32(1)]
            hip_penalty = hip_penalty + hip_dq * hip_dq
            hip_dq = joint_q[q_base + wp.int32(7) + wp.int32(2)] - default_joint_pos[wp.int32(2)]
            hip_penalty = hip_penalty + hip_dq * hip_dq
            hip_dq = joint_q[q_base + wp.int32(7) + wp.int32(7)] - default_joint_pos[wp.int32(7)]
            hip_penalty = hip_penalty + hip_dq * hip_dq
            hip_dq = joint_q[q_base + wp.int32(7) + wp.int32(8)] - default_joint_pos[wp.int32(8)]
            hip_penalty = hip_penalty + hip_dq * hip_dq
            gait_reward = gait_reward + w_gait_hip * hip_penalty

            base_dz = _clip_finite(
                joint_q[q_base + wp.int32(2)] - base_height_target, wp.float32(-10.0), wp.float32(10.0)
            )
            gait_reward = gait_reward + w_base_height * base_dz * base_dz

        biped_contact_reward = w_feet_air_time * feet_air_time_reward + w_feet_slide * feet_slide_penalty

        hip_deviation_l1 = wp.float32(0.0)
        hip_q = joint_q[q_base + wp.int32(7) + wp.int32(1)] - default_joint_pos[wp.int32(1)]
        hip_deviation_l1 = hip_deviation_l1 + wp.abs(_clip_finite(hip_q, wp.float32(-10.0), wp.float32(10.0)))
        hip_q = joint_q[q_base + wp.int32(7) + wp.int32(2)] - default_joint_pos[wp.int32(2)]
        hip_deviation_l1 = hip_deviation_l1 + wp.abs(_clip_finite(hip_q, wp.float32(-10.0), wp.float32(10.0)))
        hip_q = joint_q[q_base + wp.int32(7) + wp.int32(7)] - default_joint_pos[wp.int32(7)]
        hip_deviation_l1 = hip_deviation_l1 + wp.abs(_clip_finite(hip_q, wp.float32(-10.0), wp.float32(10.0)))
        hip_q = joint_q[q_base + wp.int32(7) + wp.int32(8)] - default_joint_pos[wp.int32(8)]
        hip_deviation_l1 = hip_deviation_l1 + wp.abs(_clip_finite(hip_q, wp.float32(-10.0), wp.float32(10.0)))

        waist_deviation_l1 = wp.float32(0.0)
        for j in range(12, 15):
            waist_q = joint_q[q_base + wp.int32(7) + j] - default_joint_pos[j]
            waist_deviation_l1 = waist_deviation_l1 + wp.abs(_clip_finite(waist_q, wp.float32(-10.0), wp.float32(10.0)))

        upper_deviation_l1 = wp.float32(0.0)
        for j in range(15, ACTION_DIM_G1):
            upper_q = joint_q[q_base + wp.int32(7) + j] - default_joint_pos[j]
            upper_deviation_l1 = upper_deviation_l1 + wp.abs(_clip_finite(upper_q, wp.float32(-10.0), wp.float32(10.0)))

        ankle_limit_penalty = wp.float32(0.0)
        for j in range(ACTION_DIM_G1):
            if j == 4 or j == 5 or j == 10 or j == 11:
                ankle_q = joint_q[q_base + wp.int32(7) + j]
                ankle_limit_penalty = ankle_limit_penalty + wp.max(ctrl_lower[j] - ankle_q, wp.float32(0.0))
                ankle_limit_penalty = ankle_limit_penalty + wp.max(ankle_q - ctrl_upper[j], wp.float32(0.0))

        joint_acc_penalty = joint_acc_l2[world]
        current_step_regularizer = episode_steps[world]
        if current_step_regularizer > joint_regularizer_episode_step[world]:
            joint_regularizer_episode_step[world] = current_step_regularizer
            joint_acc_penalty = wp.float32(0.0)
            inv_dt = wp.float32(1.0) / wp.max(reward_dt, wp.float32(1.0e-6))
            for j in range(ACTION_DIM_G1):
                qd_joint = _clip_finite(joint_qd[qd_base + wp.int32(6) + j], wp.float32(-200.0), wp.float32(200.0))
                if current_step_regularizer > wp.int32(0):
                    if j < 4 or (j >= 6 and j < 10):
                        dqd = (qd_joint - previous_joint_qd[world, j]) * inv_dt
                        joint_acc_penalty = joint_acc_penalty + dqd * dqd
                previous_joint_qd[world, j] = qd_joint
            joint_acc_l2[world] = joint_acc_penalty

        joint_regularizer_reward = (
            w_joint_deviation_hip * hip_deviation_l1
            + w_joint_deviation_waist * waist_deviation_l1
            + w_joint_deviation_upper * upper_deviation_l1
            + w_joint_acc_legs * joint_acc_penalty
            + w_joint_pos_limit_ankle * ankle_limit_penalty
        )

        sparse_success = wp.float32(0.0)
        sparse_lin_tol = wp.max(sparse_command_velocity_tolerance, wp.float32(0.0))
        sparse_yaw_tol = wp.max(sparse_command_yaw_tolerance, wp.float32(0.0))
        if vx_err * vx_err + vy_err * vy_err <= sparse_lin_tol * sparse_lin_tol:
            if wp.abs(yaw_err) <= sparse_yaw_tol:
                sparse_success = wp.float32(1.0)
        track_lin_upright = track_lin * upright_gate
        track_ang_upright = track_ang * upright_gate
        command_progress_upright = command_progress * upright_gate
        sparse_success_upright = sparse_success * upright_gate

        target_success = wp.float32(0.0)
        target_radius = wp.max(sparse_target_radius, wp.float32(0.0))
        target_dist_sq = target_delta_w[0] * target_delta_w[0] + target_delta_w[1] * target_delta_w[1]
        target_progress = wp.float32(0.0)
        if target_dist_sq > wp.float32(1.0e-8):
            inv_target_dist = wp.float32(1.0) / wp.sqrt(target_dist_sq)
            target_progress = (
                target_delta_w[0] * lin_origin_w[0] + target_delta_w[1] * lin_origin_w[1]
            ) * inv_target_dist
        target_progress_upright = target_progress * upright_gate
        target_success_min_height = wp.max(sparse_target_success_min_base_height, wp.float32(0.0))
        target_success_max_height = wp.max(sparse_target_success_max_base_height, target_success_min_height)
        target_success_upright = _clip_float(sparse_target_success_upright_cos, wp.float32(-1.0), wp.float32(1.0))
        if target_dist_sq <= target_radius * target_radius:
            if upright_cos >= target_success_upright:
                if joint_q[q_base + wp.int32(2)] >= target_success_min_height:
                    if joint_q[q_base + wp.int32(2)] <= target_success_max_height:
                        target_success = wp.float32(1.0)

        fall = wp.float32(0.0)
        if joint_q[q_base + wp.int32(2)] < min_base_height or upright_cos < min_upright_cos:
            fall = wp.float32(1.0)
        if state_bad != wp.int32(0):
            fall = wp.float32(1.0)

        reward = wp.float32(0.0)
        success_metric = track_lin_upright
        if reward_mode == wp.int32(0):
            # nanoG1 keeps tracking active during recovery; the upright gate only
            # delays torso-stability penalties until the policy is competent.
            success_metric = track_lin
            shaped_reward = (
                w_track_lin * track_lin
                + w_track_ang * track_ang
                + w_command_progress * command_progress
                + w_lin_vel_z * lin_vel_z_penalty
                + w_ang_vel_xy * ang_vel_xy_penalty * upright_gate
                + w_orientation * orientation_penalty * upright_gate
                + w_torque * torque_sq_penalty
                + w_action_rate * action_rate_penalty
                + gait_reward
                + biped_contact_reward
                + joint_regularizer_reward
                + w_alive
            )
            reward = shaped_reward * reward_dt
        elif reward_mode == wp.int32(1):
            shaped_reward = (
                w_sparse_command_success * sparse_success_upright
                + w_mechanical_power * mechanical_power_penalty
                + biped_contact_reward
                + joint_regularizer_reward
            )
            reward = shaped_reward * reward_dt
            success_metric = sparse_success_upright
        elif reward_mode == wp.int32(2):
            reward = (
                w_sparse_command_success * target_success
                + (
                    w_target_progress * target_progress_upright
                    + w_mechanical_power * mechanical_power_penalty
                    + biped_contact_reward
                    + joint_regularizer_reward
                )
                * reward_dt
            )
            success_metric = target_success
        elif reward_mode == wp.int32(4):
            shaped_reward = (
                w_target_progress * target_progress_upright
                + w_track_ang * track_ang_upright
                + w_lin_vel_z * lin_vel_z_penalty
                + w_ang_vel_xy * ang_vel_xy_penalty * upright_gate
                + w_orientation * orientation_penalty * upright_gate
                + w_torque * torque_sq_penalty
                + w_action_rate * action_rate_penalty
                + gait_reward
                + biped_contact_reward
                + joint_regularizer_reward
                + w_alive
                + w_mechanical_power * mechanical_power_penalty
            )
            reward = w_sparse_command_success * target_success + shaped_reward * reward_dt
            success_metric = target_success
        else:
            shaped_reward = (
                w_track_lin * track_lin_upright
                + w_track_ang * track_ang_upright
                + w_command_progress * command_progress_upright
                + w_lin_vel_z * lin_vel_z_penalty
                + w_ang_vel_xy * ang_vel_xy_penalty * upright_gate
                + w_orientation * orientation_penalty * upright_gate
                + w_torque * torque_sq_penalty
                + w_action_rate * action_rate_penalty
                + gait_reward
                + biped_contact_reward
                + joint_regularizer_reward
                + w_alive
                + w_sparse_command_success * sparse_success_upright
                + w_mechanical_power * mechanical_power_penalty
            )
            reward = shaped_reward * reward_dt
            success_metric = sparse_success_upright
        if fall > wp.float32(0.5):
            reward = reward + w_termination
            success_metric = wp.float32(0.0)
        if state_bad != wp.int32(0) or not wp.isfinite(reward):
            reward = w_termination
            success_metric = wp.float32(0.0)
        rewards[world] = reward
        successes[world] = success_metric

        done = wp.float32(0.0)
        if fall > wp.float32(0.5):
            done = wp.float32(1.0)
        if (reward_mode == wp.int32(2) or reward_mode == wp.int32(4)) and target_success > wp.float32(0.5):
            done = wp.float32(1.0)
        if max_episode_steps > wp.int32(0):
            if episode_steps[world] >= max_episode_steps:
                done = wp.float32(1.0)
        dones[world] = done


@wp.kernel
def g1_increment_episode_steps_kernel(episode_steps: wp.array[wp.int32]):
    world = wp.tid()
    episode_steps[world] = episode_steps[world] + wp.int32(1)


@wp.kernel
def g1_reset_done_worlds_kernel(
    seed: wp.int32,
    reset_noise: wp.float32,
    dones: wp.array[wp.float32],
    default_joint_q: wp.array[wp.float32],
    default_joint_qd: wp.array[wp.float32],
    coord_stride: wp.int32,
    dof_stride: wp.int32,
    action_dim: wp.int32,
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    episode_steps: wp.array[wp.int32],
    previous_actions: wp.array2d[wp.float32],
    current_actions: wp.array2d[wp.float32],
    actuator_force: wp.array2d[wp.float32],
    previous_joint_qd: wp.array2d[wp.float32],
    joint_acc_l2: wp.array[wp.float32],
    joint_regularizer_episode_step: wp.array[wp.int32],
    foot_air_time: wp.array2d[wp.float32],
    foot_contact_time: wp.array2d[wp.float32],
    foot_timer_episode_step: wp.array[wp.int32],
    joint_f: wp.array[wp.float32],
    reset_articulation_mask: wp.array[wp.bool],
):
    world, col = wp.tid()
    should_reset = dones[world] > wp.float32(0.5)
    if col == 0:
        reset_articulation_mask[world] = should_reset
    if not should_reset:
        return
    if col < coord_stride:
        idx = world * coord_stride + col
        value = default_joint_q[idx]
        if col >= wp.int32(7) and col < wp.int32(7) + action_dim and reset_noise > wp.float32(0.0):
            rng = wp.rand_init(seed, world * coord_stride + col)
            value = value + reset_noise * (wp.float32(2.0) * wp.randf(rng) - wp.float32(1.0))
        joint_q[idx] = value
    if col < dof_stride:
        idx = world * dof_stride + col
        joint_qd[idx] = default_joint_qd[idx]
        joint_f[idx] = wp.float32(0.0)
    if col < action_dim:
        previous_actions[world, col] = wp.float32(0.0)
        current_actions[world, col] = wp.float32(0.0)
        actuator_force[world, col] = wp.float32(0.0)
        previous_joint_qd[world, col] = default_joint_qd[world * dof_stride + wp.int32(6) + col]
    if col < wp.int32(2):
        foot_air_time[world, col] = wp.float32(0.0)
        foot_contact_time[world, col] = wp.float32(0.0)
    if col == 0:
        episode_steps[world] = wp.int32(0)
        foot_timer_episode_step[world] = wp.int32(-1)
        joint_acc_l2[world] = wp.float32(0.0)
        joint_regularizer_episode_step[world] = wp.int32(-1)


@wp.kernel
def g1_reset_done_worlds_seed_counter_kernel(
    seed_counter: wp.array[wp.int32],
    seed_offset: wp.int32,
    reset_noise: wp.float32,
    dones: wp.array[wp.float32],
    default_joint_q: wp.array[wp.float32],
    default_joint_qd: wp.array[wp.float32],
    coord_stride: wp.int32,
    dof_stride: wp.int32,
    action_dim: wp.int32,
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    episode_steps: wp.array[wp.int32],
    previous_actions: wp.array2d[wp.float32],
    current_actions: wp.array2d[wp.float32],
    actuator_force: wp.array2d[wp.float32],
    previous_joint_qd: wp.array2d[wp.float32],
    joint_acc_l2: wp.array[wp.float32],
    joint_regularizer_episode_step: wp.array[wp.int32],
    foot_air_time: wp.array2d[wp.float32],
    foot_contact_time: wp.array2d[wp.float32],
    foot_timer_episode_step: wp.array[wp.int32],
    joint_f: wp.array[wp.float32],
    reset_articulation_mask: wp.array[wp.bool],
):
    world, col = wp.tid()
    should_reset = dones[world] > wp.float32(0.5)
    if col == 0:
        reset_articulation_mask[world] = should_reset
    if not should_reset:
        return
    seed = wp.int32((wp.int64(seed_counter[0]) + wp.int64(seed_offset)) % wp.int64(2147483647))
    if col < coord_stride:
        idx = world * coord_stride + col
        value = default_joint_q[idx]
        if col >= wp.int32(7) and col < wp.int32(7) + action_dim and reset_noise > wp.float32(0.0):
            rng = wp.rand_init(seed, world * coord_stride + col)
            value = value + reset_noise * (wp.float32(2.0) * wp.randf(rng) - wp.float32(1.0))
        joint_q[idx] = value
    if col < dof_stride:
        idx = world * dof_stride + col
        joint_qd[idx] = default_joint_qd[idx]
        joint_f[idx] = wp.float32(0.0)
    if col < action_dim:
        previous_actions[world, col] = wp.float32(0.0)
        current_actions[world, col] = wp.float32(0.0)
        actuator_force[world, col] = wp.float32(0.0)
        previous_joint_qd[world, col] = default_joint_qd[world * dof_stride + wp.int32(6) + col]
    if col < wp.int32(2):
        foot_air_time[world, col] = wp.float32(0.0)
        foot_contact_time[world, col] = wp.float32(0.0)
    if col == 0:
        episode_steps[world] = wp.int32(0)
        foot_timer_episode_step[world] = wp.int32(-1)
        joint_acc_l2[world] = wp.float32(0.0)
        joint_regularizer_episode_step[world] = wp.int32(-1)


@wp.func
def _g1_sample_command_value(
    seed: wp.int32,
    world: wp.int32,
    col: wp.int32,
    x_min: wp.float32,
    x_max: wp.float32,
    y_min: wp.float32,
    y_max: wp.float32,
    yaw_min: wp.float32,
    yaw_max: wp.float32,
    command_scale: wp.float32,
    zero_probability: wp.float32,
):
    zero_rng = wp.rand_init(seed, world * wp.int32(4))
    if zero_probability > wp.float32(0.0) and wp.randf(zero_rng) < zero_probability:
        return wp.float32(0.0)

    rng = wp.rand_init(seed, world * wp.int32(4) + col + wp.int32(1))
    u = wp.randf(rng)
    value = yaw_min + (yaw_max - yaw_min) * u
    if col == wp.int32(0):
        value = x_min + (x_max - x_min) * u
    elif col == wp.int32(1):
        value = y_min + (y_max - y_min) * u
    return command_scale * value


@wp.kernel
def g1_sample_commands_kernel(
    seed: wp.int32,
    x_min: wp.float32,
    x_max: wp.float32,
    y_min: wp.float32,
    y_max: wp.float32,
    yaw_min: wp.float32,
    yaw_max: wp.float32,
    command_scale: wp.array[wp.float32],
    zero_probability: wp.float32,
    command: wp.array2d[wp.float32],
):
    world, col = wp.tid()
    command[world, col] = _g1_sample_command_value(
        seed, world, col, x_min, x_max, y_min, y_max, yaw_min, yaw_max, command_scale[0], zero_probability
    )


@wp.kernel
def g1_sample_commands_seed_counter_kernel(
    seed_counter: wp.array[wp.int32],
    seed_offset: wp.int32,
    x_min: wp.float32,
    x_max: wp.float32,
    y_min: wp.float32,
    y_max: wp.float32,
    yaw_min: wp.float32,
    yaw_max: wp.float32,
    command_scale: wp.array[wp.float32],
    zero_probability: wp.float32,
    command: wp.array2d[wp.float32],
):
    world, col = wp.tid()
    seed = wp.int32((wp.int64(seed_counter[0]) + wp.int64(seed_offset)) % wp.int64(2147483647))
    command[world, col] = _g1_sample_command_value(
        seed, world, col, x_min, x_max, y_min, y_max, yaw_min, yaw_max, command_scale[0], zero_probability
    )


@wp.kernel
def g1_sample_done_commands_kernel(
    seed: wp.int32,
    x_min: wp.float32,
    x_max: wp.float32,
    y_min: wp.float32,
    y_max: wp.float32,
    yaw_min: wp.float32,
    yaw_max: wp.float32,
    command_scale: wp.array[wp.float32],
    zero_probability: wp.float32,
    dones: wp.array[wp.float32],
    command: wp.array2d[wp.float32],
):
    world, col = wp.tid()
    if dones[world] <= wp.float32(0.5):
        return
    command[world, col] = _g1_sample_command_value(
        seed, world, col, x_min, x_max, y_min, y_max, yaw_min, yaw_max, command_scale[0], zero_probability
    )


@wp.kernel
def g1_sample_done_commands_seed_counter_kernel(
    seed_counter: wp.array[wp.int32],
    seed_offset: wp.int32,
    x_min: wp.float32,
    x_max: wp.float32,
    y_min: wp.float32,
    y_max: wp.float32,
    yaw_min: wp.float32,
    yaw_max: wp.float32,
    command_scale: wp.array[wp.float32],
    zero_probability: wp.float32,
    dones: wp.array[wp.float32],
    command: wp.array2d[wp.float32],
):
    world, col = wp.tid()
    if dones[world] <= wp.float32(0.5):
        return
    seed = wp.int32((wp.int64(seed_counter[0]) + wp.int64(seed_offset)) % wp.int64(2147483647))
    command[world, col] = _g1_sample_command_value(
        seed, world, col, x_min, x_max, y_min, y_max, yaw_min, yaw_max, command_scale[0], zero_probability
    )


@wp.kernel
def g1_sample_due_commands_kernel(
    seed: wp.int32,
    resample_steps: wp.int32,
    x_min: wp.float32,
    x_max: wp.float32,
    y_min: wp.float32,
    y_max: wp.float32,
    yaw_min: wp.float32,
    yaw_max: wp.float32,
    command_scale: wp.array[wp.float32],
    zero_probability: wp.float32,
    dones: wp.array[wp.float32],
    episode_steps: wp.array[wp.int32],
    command: wp.array2d[wp.float32],
):
    world, col = wp.tid()
    if resample_steps <= wp.int32(0) or dones[world] > wp.float32(0.5):
        return
    if episode_steps[world] <= wp.int32(0) or episode_steps[world] % resample_steps != wp.int32(0):
        return
    command[world, col] = _g1_sample_command_value(
        seed, world, col, x_min, x_max, y_min, y_max, yaw_min, yaw_max, command_scale[0], zero_probability
    )


@wp.kernel
def g1_sample_due_commands_seed_counter_kernel(
    seed_counter: wp.array[wp.int32],
    seed_offset: wp.int32,
    resample_steps: wp.int32,
    x_min: wp.float32,
    x_max: wp.float32,
    y_min: wp.float32,
    y_max: wp.float32,
    yaw_min: wp.float32,
    yaw_max: wp.float32,
    command_scale: wp.array[wp.float32],
    zero_probability: wp.float32,
    dones: wp.array[wp.float32],
    episode_steps: wp.array[wp.int32],
    command: wp.array2d[wp.float32],
):
    world, col = wp.tid()
    if resample_steps <= wp.int32(0) or dones[world] > wp.float32(0.5):
        return
    if episode_steps[world] <= wp.int32(0) or episode_steps[world] % resample_steps != wp.int32(0):
        return
    seed = wp.int32((wp.int64(seed_counter[0]) + wp.int64(seed_offset)) % wp.int64(2147483647))
    command[world, col] = _g1_sample_command_value(
        seed, world, col, x_min, x_max, y_min, y_max, yaw_min, yaw_max, command_scale[0], zero_probability
    )


@wp.kernel(enable_backward=False)
def g1_update_command_scale_kernel(
    sample_counter: wp.array[wp.int32],
    sample_delta: wp.int32,
    start_scale: wp.float32,
    ramp_samples: wp.float32,
    command_scale: wp.array[wp.float32],
):
    next_count = sample_counter[0] + sample_delta
    if next_count < sample_counter[0]:
        next_count = sample_counter[0]
    sample_counter[0] = next_count

    start = _clip_float(start_scale, wp.float32(0.0), wp.float32(1.0))
    scale = wp.float32(1.0)
    if ramp_samples > wp.float32(0.0) and start < wp.float32(1.0):
        progress = wp.float32(next_count) / ramp_samples
        progress = _clip_float(progress, wp.float32(0.0), wp.float32(1.0))
        scale = start + (wp.float32(1.0) - start) * progress
    command_scale[0] = scale


@wp.kernel(enable_backward=False)
def g1_update_sparse_target_distance_kernel(
    sample_counter: wp.array[wp.int32],
    sample_delta: wp.int32,
    start_distance: wp.float32,
    end_distance: wp.float32,
    ramp_samples: wp.float32,
    target_distance: wp.array[wp.float32],
):
    next_count = sample_counter[0] + sample_delta
    if next_count < sample_counter[0]:
        next_count = sample_counter[0]
    sample_counter[0] = next_count

    start = wp.max(start_distance, wp.float32(0.0))
    end = wp.max(end_distance, wp.float32(0.0))
    distance = end
    if ramp_samples > wp.float32(0.0):
        progress = wp.float32(next_count) / ramp_samples
        progress = _clip_float(progress, wp.float32(0.0), wp.float32(1.0))
        distance = start + (end - start) * progress
    target_distance[0] = distance


@wp.kernel
def g1_set_sparse_targets_from_distance_kernel(
    base_x: wp.float32,
    base_y: wp.float32,
    target_distance: wp.array[wp.float32],
    target_position: wp.array2d[wp.float32],
):
    world, col = wp.tid()
    norm = wp.sqrt(base_x * base_x + base_y * base_y)
    dir_x = wp.float32(1.0)
    dir_y = wp.float32(0.0)
    if norm > wp.float32(1.0e-6):
        inv_norm = wp.float32(1.0) / norm
        dir_x = base_x * inv_norm
        dir_y = base_y * inv_norm
    value = target_distance[0] * dir_x
    if col == wp.int32(1):
        value = target_distance[0] * dir_y
    target_position[world, col] = value


@wp.kernel
def g1_sample_sparse_targets_seed_counter_kernel(
    seed_counter: wp.array[wp.int32],
    seed_offset: wp.int32,
    base_angle: wp.float32,
    angle_min: wp.float32,
    angle_max: wp.float32,
    target_distance: wp.array[wp.float32],
    target_position: wp.array2d[wp.float32],
):
    world, col = wp.tid()
    seed = wp.int32((wp.int64(seed_counter[0]) + wp.int64(seed_offset)) % wp.int64(2147483647))
    rng = wp.rand_init(seed, world)
    angle = base_angle + angle_min + (angle_max - angle_min) * wp.randf(rng)
    value = target_distance[0] * wp.cos(angle)
    if col == wp.int32(1):
        value = target_distance[0] * wp.sin(angle)
    target_position[world, col] = value


@wp.kernel
def g1_sample_sparse_target_range_seed_counter_kernel(
    seed_counter: wp.array[wp.int32],
    seed_offset: wp.int32,
    base_angle: wp.float32,
    angle_min: wp.float32,
    angle_max: wp.float32,
    min_distance: wp.float32,
    target_distance: wp.array[wp.float32],
    target_position: wp.array2d[wp.float32],
):
    world, col = wp.tid()
    seed = wp.int32((wp.int64(seed_counter[0]) + wp.int64(seed_offset)) % wp.int64(2147483647))
    rng = wp.rand_init(seed, world)
    max_distance = wp.max(target_distance[0], wp.float32(0.0))
    lo = _clip_float(min_distance, wp.float32(0.0), max_distance)
    distance = lo + (max_distance - lo) * wp.randf(rng)
    angle = base_angle + angle_min + (angle_max - angle_min) * wp.randf(rng)
    value = distance * wp.cos(angle)
    if col == wp.int32(1):
        value = distance * wp.sin(angle)
    target_position[world, col] = value


@wp.func
def _g1_mark_foot_contact(
    shape_id: wp.int32,
    shape_body: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    body_stride: wp.int32,
    left_foot_body: wp.int32,
    right_foot_body: wp.int32,
    foot_contacts: wp.array2d[wp.float32],
):
    if shape_id < wp.int32(0):
        return
    body = shape_body[shape_id]
    if body < wp.int32(0):
        return
    world = shape_world[shape_id]
    if world < wp.int32(0):
        if body_stride <= wp.int32(0):
            return
        world = body / body_stride
    if world < wp.int32(0) or world >= foot_contacts.shape[0]:
        return
    local_body = body
    if body_stride > wp.int32(0):
        local_body = body - world * body_stride
    if local_body == left_foot_body:
        wp.atomic_add(foot_contacts, world, wp.int32(0), wp.float32(1.0))
    elif local_body == right_foot_body:
        wp.atomic_add(foot_contacts, world, wp.int32(1), wp.float32(1.0))


@wp.kernel
def g1_scan_foot_contacts_kernel(
    rigid_contact_count: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    contact_container: ContactContainer,
    body_stride: wp.int32,
    left_foot_body: wp.int32,
    right_foot_body: wp.int32,
    normal_impulse_threshold: wp.float32,
    foot_contacts: wp.array2d[wp.float32],
):
    tid = wp.tid()
    count = rigid_contact_count[0]
    if count > rigid_contact_shape0.shape[0]:
        count = rigid_contact_shape0.shape[0]
    if tid >= count:
        return
    if wp.max(cc_get_normal_lambda(contact_container, tid), wp.float32(0.0)) <= normal_impulse_threshold:
        return
    _g1_mark_foot_contact(
        rigid_contact_shape0[tid], shape_body, shape_world, body_stride, left_foot_body, right_foot_body, foot_contacts
    )
    _g1_mark_foot_contact(
        rigid_contact_shape1[tid], shape_body, shape_world, body_stride, left_foot_body, right_foot_body, foot_contacts
    )


@wp.kernel(enable_backward=False)
def g1_gather_actuator_force_kernel(
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    joint_target_q: wp.array[wp.float32],
    actuator_force_kp: wp.array[wp.float32],
    actuator_force_kd: wp.array[wp.float32],
    passive_damping: wp.array[wp.float32],
    actuator_force_lower: wp.array[wp.float32],
    actuator_force_upper: wp.array[wp.float32],
    coord_stride: wp.int32,
    dof_stride: wp.int32,
    target_uses_coord_layout: wp.int32,
    scatter_joint_f: wp.int32,
    actuator_force: wp.array2d[wp.float32],
    joint_f: wp.array[wp.float32],
):
    world, action = wp.tid()
    q = joint_q[world * coord_stride + wp.int32(7) + action]
    qd = joint_qd[world * dof_stride + wp.int32(6) + action]
    if target_uses_coord_layout != wp.int32(0):
        target = joint_target_q[world * coord_stride + wp.int32(7) + action]
    else:
        target = joint_target_q[world * dof_stride + wp.int32(6) + action]
    force = actuator_force_kp[action] * (target - q) - actuator_force_kd[action] * qd
    force = _clip_float(force, actuator_force_lower[action], actuator_force_upper[action])
    actuator_force[world, action] = force
    if scatter_joint_f != wp.int32(0):
        joint_f[world * dof_stride + wp.int32(6) + action] = force - passive_damping[action] * qd


@dataclass
class ConfigEnvG1PhoenX:
    """Configuration for :class:`EnvG1PhoenX`.

    Args:
        world_count: Number of vectorized G1 worlds.
        frame_dt: Policy step duration [s].
        sim_substeps: PhoenX physics steps per policy step. The G1
            environment owns this decimation loop and keeps the internal
            SolverPhoenX substep count at one.
        solver_iterations: PhoenX position iterations per substep.
        velocity_iterations: PhoenX velocity iterations per substep.
        joint_friction_model: Solver joint-friction mode ("hard" or "mujoco").
        joint_friction_scale: Dimensionless scale applied to nanoG1 joint friction.
        actuation_model: ``"explicit_torque"`` applies nanoG1/MuJoCo-style
            clamped PD torques through ``control.joint_f``.
            ``"constraint_drive"`` uses PhoenX implicit drive rows.
        action_scale: Position target scale [rad].
        controlled_action_count: Number of leading policy actions applied to joints.
        command: Target body-frame command ``(vx, vy, yaw_rate)`` [m/s, m/s, rad/s].
        command_x_range: Sampled command-x range [m/s].
        command_y_range: Sampled command-y range [m/s].
        command_yaw_range: Sampled yaw-rate command range [rad/s].
        randomize_commands_on_reset: Sample commands when worlds reset.
        command_zero_probability: Probability of sampling a zero command.
        command_resample_steps: Periodic command resampling interval in policy
            steps. Use ``0`` to disable periodic resampling.
        max_episode_steps: Episode timeout in policy steps. Use ``0`` to disable.
        reset_noise: Uniform joint-position reset noise half-width [rad].
        min_base_height: Episode ends below this base height [m].
        min_upright_cos: Episode ends below this base-upright cosine threshold.
        max_abs_root_position: Episode ends if any free-root coordinate exceeds
            this absolute world-space position limit [m]. Use ``0`` to disable.
        max_abs_root_linear_velocity: Episode ends if root linear velocity
            exceeds this magnitude [m/s]. Use ``0`` to disable.
        max_abs_root_angular_velocity: Episode ends if root angular velocity
            exceeds this magnitude [rad/s]. Use ``0`` to disable.
        max_abs_joint_position: Episode ends if any actuated joint coordinate
            exceeds this absolute angle limit [rad]. Use ``0`` to disable.
        max_abs_joint_velocity: Episode ends if any actuated joint velocity
            exceeds this absolute speed limit [rad/s]. Use ``0`` to disable.
        phase_period: Gait clock period in policy steps.
        observation_mode: Observation layout. ``"nanog1"`` keeps the
            nanoG1 deploy contract. ``"isaaclab_flat"`` uses the IsaacLab
            flat-locomotion layout with body-frame base linear velocity and no
            phase clock.
        w_track_lin: Linear XY velocity tracking reward scale.
        w_track_ang: Yaw-rate tracking reward scale.
        w_command_progress: Command-aligned velocity projection reward scale.
        w_lin_vel_z: Vertical body-velocity penalty scale.
        w_ang_vel_xy: Base roll/pitch angular-velocity penalty scale.
        w_orientation: Projected-gravity tilt penalty scale.
        w_torque: Torque-squared penalty scale.
        w_action_rate: Action-rate penalty scale.
        w_alive: Alive reward per policy step.
        w_termination: Termination reward applied on fall.
        reward_mode: Reward mode, either ``"nanog1_dense"``, ``"sparse_command"``,
            ``"sparse_target"``, or ``"dense_sparse_command"``.
        w_sparse_command_success: Sparse command or target success reward scale.
        w_target_progress: Sparse-target progress velocity reward scale.
        sparse_command_velocity_tolerance: Linear command-success tolerance [m/s].
        sparse_command_yaw_tolerance: Yaw-rate command-success tolerance [rad/s].
        sparse_target_position: Sparse target XY position in each world [m].
        sparse_target_radius: Sparse target success radius [m].
        sparse_target_success_upright_cos: Minimum upright cosine for sparse target success.
        sparse_target_success_min_base_height: Minimum base height for sparse target success [m].
        sparse_target_success_max_base_height: Maximum base height for sparse target success [m].
        w_mechanical_power: Absolute joint mechanical-power penalty scale.
        gait_stance_fraction: Fraction of each gait half-period spent in stance.
        w_gait_contact: Stance/contact phase-matching reward scale.
        w_gait_swing: Swing-foot height penalty scale.
        w_gait_swing_contact: Penalty for a swing foot that remains in contact.
        w_gait_hip: Hip-roll/yaw deviation penalty scale.
        gait_foot_height: Swing-foot target height [m].
        w_base_height: Base-height penalty scale.
        base_height_target: Target base height [m].
        w_feet_air_time: IsaacLab-style positive biped feet-air-time reward scale.
        feet_air_time_threshold: Saturation time for feet-air-time reward [s].
        w_feet_slide: Contact foot XY sliding penalty scale.
        foot_contact_normal_impulse_threshold: Minimum accumulated normal
            impulse [N*s] for a foot contact to count as active.
        w_joint_deviation_hip: Hip yaw/roll joint-deviation penalty scale.
        w_joint_deviation_waist: Waist joint-deviation penalty scale.
        w_joint_deviation_upper: Upper-body joint-deviation penalty scale.
        w_joint_acc_legs: Leg joint-acceleration penalty scale.
        w_joint_pos_limit_ankle: Ankle position-limit violation penalty scale.
        parse_meshes: Import MJCF mesh collision geoms. Disable for fast
            RL runs that use the primitive foot and arm geoms only.
        parse_visuals: Import visual-only MJCF meshes for rendering. When this
            is enabled with ``parse_meshes=False``, G1 mesh colliders are
            filtered out so contact behavior stays on the primitive/nanoG1
            collision recipe.
        contact_geometry: G1 contact geometry preset. "mjcf" keeps the
            imported MJCF primitives; "nanog1_foot_boxes" replaces the
            four foot point contacts per foot with nanoG1's MuJoCo contact boxes.
        ground_friction: Ground-plane Coulomb friction coefficient.
        foot_box_xy_scale: Scale applied to the nanoG1 foot-box X/Y half-extents.
        auto_reset: Reset worlds whose done flag is set after each step.
        rigid_contact_max_per_world: Rigid-contact capacity per vectorized world.
            ``0`` keeps the solver's automatic sizing. The default is
            intended for primitive-collision G1 RL runs and is ignored when
            ``parse_meshes`` is enabled.
        threads_per_world: PhoenX multi-world lane count, or ``"auto"``.
        multi_world_scheduler: PhoenX multi-world scheduler selection.
        prepare_refresh_stride: PhoenX cached-prepare refresh stride, or
            ``"auto"``.
    """

    world_count: int = g1_recipe.WORLD_COUNT
    frame_dt: float = g1_recipe.FRAME_DT
    sim_substeps: int = g1_recipe.SIM_SUBSTEPS
    solver_iterations: int = g1_recipe.SOLVER_ITERATIONS
    velocity_iterations: int = g1_recipe.VELOCITY_ITERATIONS
    joint_friction_model: str = g1_recipe.JOINT_FRICTION_MODEL
    joint_friction_scale: float = g1_recipe.JOINT_FRICTION_SCALE
    actuation_model: str = g1_recipe.ACTUATION_MODEL
    action_scale: float = g1_recipe.ACTION_SCALE
    controlled_action_count: int = g1_recipe.CONTROLLED_ACTION_COUNT
    command: tuple[float, float, float] = g1_recipe.COMMAND
    command_x_range: tuple[float, float] = g1_recipe.COMMAND_X_RANGE
    command_y_range: tuple[float, float] = g1_recipe.COMMAND_Y_RANGE
    command_yaw_range: tuple[float, float] = g1_recipe.COMMAND_YAW_RANGE
    randomize_commands_on_reset: bool = False
    command_zero_probability: float = 0.0
    command_resample_steps: int = 0
    max_episode_steps: int = g1_recipe.MAX_EPISODE_STEPS
    reset_noise: float = g1_recipe.RESET_NOISE
    min_base_height: float = g1_recipe.MIN_BASE_HEIGHT
    min_upright_cos: float = g1_recipe.MIN_UPRIGHT_COS
    max_abs_root_position: float = g1_recipe.MAX_ABS_ROOT_POSITION
    max_abs_root_linear_velocity: float = g1_recipe.MAX_ABS_ROOT_LINEAR_VELOCITY
    max_abs_root_angular_velocity: float = g1_recipe.MAX_ABS_ROOT_ANGULAR_VELOCITY
    max_abs_joint_position: float = g1_recipe.MAX_ABS_JOINT_POSITION
    max_abs_joint_velocity: float = g1_recipe.MAX_ABS_JOINT_VELOCITY
    phase_period: int = g1_recipe.PHASE_PERIOD
    observation_mode: str = g1_recipe.OBSERVATION_MODE
    w_track_lin: float = g1_recipe.W_TRACK_LIN
    w_track_ang: float = g1_recipe.W_TRACK_ANG
    w_command_progress: float = g1_recipe.W_COMMAND_PROGRESS
    w_lin_vel_z: float = g1_recipe.W_LIN_VEL_Z
    w_ang_vel_xy: float = g1_recipe.W_ANG_VEL_XY
    w_orientation: float = g1_recipe.W_ORIENTATION
    w_torque: float = g1_recipe.W_TORQUE
    w_action_rate: float = g1_recipe.W_ACTION_RATE
    w_alive: float = g1_recipe.W_ALIVE
    w_termination: float = g1_recipe.W_TERMINATION
    reward_mode: str = g1_recipe.REWARD_MODE
    w_sparse_command_success: float = g1_recipe.W_SPARSE_COMMAND_SUCCESS
    w_target_progress: float = g1_recipe.W_TARGET_PROGRESS
    sparse_command_velocity_tolerance: float = g1_recipe.SPARSE_COMMAND_VELOCITY_TOLERANCE
    sparse_command_yaw_tolerance: float = g1_recipe.SPARSE_COMMAND_YAW_TOLERANCE
    sparse_target_position: tuple[float, float] = g1_recipe.SPARSE_TARGET_POSITION
    sparse_target_radius: float = g1_recipe.SPARSE_TARGET_RADIUS
    sparse_target_success_upright_cos: float = g1_recipe.SPARSE_TARGET_SUCCESS_UPRIGHT_COS
    sparse_target_success_min_base_height: float = g1_recipe.SPARSE_TARGET_SUCCESS_MIN_BASE_HEIGHT
    sparse_target_success_max_base_height: float = g1_recipe.SPARSE_TARGET_SUCCESS_MAX_BASE_HEIGHT
    w_mechanical_power: float = g1_recipe.W_MECHANICAL_POWER
    gait_stance_fraction: float = g1_recipe.GAIT_STANCE_FRACTION
    w_gait_contact: float = g1_recipe.W_GAIT_CONTACT
    w_gait_swing: float = g1_recipe.W_GAIT_SWING
    w_gait_swing_contact: float = g1_recipe.W_GAIT_SWING_CONTACT
    w_gait_hip: float = g1_recipe.W_GAIT_HIP
    gait_foot_height: float = g1_recipe.GAIT_FOOT_HEIGHT
    w_base_height: float = g1_recipe.W_BASE_HEIGHT
    base_height_target: float = g1_recipe.BASE_HEIGHT_TARGET
    w_feet_air_time: float = g1_recipe.W_FEET_AIR_TIME
    feet_air_time_threshold: float = g1_recipe.FEET_AIR_TIME_THRESHOLD
    w_feet_slide: float = g1_recipe.W_FEET_SLIDE
    foot_contact_normal_impulse_threshold: float = g1_recipe.FOOT_CONTACT_NORMAL_IMPULSE_THRESHOLD
    w_joint_deviation_hip: float = g1_recipe.W_JOINT_DEVIATION_HIP
    w_joint_deviation_waist: float = g1_recipe.W_JOINT_DEVIATION_WAIST
    w_joint_deviation_upper: float = g1_recipe.W_JOINT_DEVIATION_UPPER
    w_joint_acc_legs: float = g1_recipe.W_JOINT_ACC_LEGS
    w_joint_pos_limit_ankle: float = g1_recipe.W_JOINT_POS_LIMIT_ANKLE
    parse_meshes: bool = g1_recipe.PARSE_MESHES
    parse_visuals: bool = g1_recipe.PARSE_VISUALS
    contact_geometry: str = g1_recipe.CONTACT_GEOMETRY
    ground_friction: float = g1_recipe.GROUND_FRICTION
    foot_box_xy_scale: float = g1_recipe.FOOT_BOX_XY_SCALE
    auto_reset: bool = g1_recipe.AUTO_RESET
    rigid_contact_max_per_world: int = g1_recipe.RIGID_CONTACT_MAX_PER_WORLD
    threads_per_world: int | str = g1_recipe.THREADS_PER_WORLD
    multi_world_scheduler: str = g1_recipe.MULTI_WORLD_SCHEDULER
    prepare_refresh_stride: int | str = g1_recipe.PREPARE_REFRESH_STRIDE
    articulation_mode: str = "maximal"


class EnvG1PhoenX:
    """Warp-only Unitree G1 locomotion environment backed by SolverPhoenX.

    The observation layout follows nanoG1 v3: scaled base angular velocity,
    projected gravity, command, joint position deltas, scaled joint velocities,
    previous actions, and a sin/cos gait phase clock.

    Args:
        config: Environment and reward configuration.
        device: Warp device.
    """

    obs_dim = OBS_DIM_G1
    action_dim = ACTION_DIM_G1

    def __init__(self, config: ConfigEnvG1PhoenX | None = None, *, device: wp.context.Devicelike = None):
        self.config = config or ConfigEnvG1PhoenX()
        self.device = wp.get_device(device)
        self.world_count = int(self.config.world_count)
        if self.world_count <= 0:
            raise ValueError("world_count must be positive")
        if not 0 < int(self.config.controlled_action_count) <= ACTION_DIM_G1:
            raise ValueError("controlled_action_count must be in [1, ACTION_DIM_G1]")
        if int(self.config.phase_period) <= 0:
            raise ValueError("phase_period must be positive")
        if int(self.config.rigid_contact_max_per_world) < 0:
            raise ValueError("rigid_contact_max_per_world must be non-negative")
        if str(self.config.joint_friction_model) not in ("hard", "mujoco"):
            raise ValueError('joint_friction_model must be "hard" or "mujoco"')
        if float(self.config.joint_friction_scale) < 0.0:
            raise ValueError("joint_friction_scale must be non-negative")
        if str(self.config.actuation_model) not in ("explicit_torque", "constraint_drive"):
            raise ValueError('actuation_model must be "explicit_torque" or "constraint_drive"')
        self._check_command_ranges(
            self.config.command_x_range, self.config.command_y_range, self.config.command_yaw_range
        )
        if not 0.0 <= float(self.config.command_zero_probability) <= 1.0:
            raise ValueError("command_zero_probability must be in [0, 1]")
        if int(self.config.command_resample_steps) < 0:
            raise ValueError("command_resample_steps must be non-negative")
        if not 0.0 <= float(self.config.gait_stance_fraction) <= 1.0:
            raise ValueError("gait_stance_fraction must be in [0, 1]")
        if float(self.config.sparse_target_radius) < 0.0:
            raise ValueError("sparse_target_radius must be non-negative")
        if float(self.config.feet_air_time_threshold) < 0.0:
            raise ValueError("feet_air_time_threshold must be non-negative")
        if float(self.config.foot_contact_normal_impulse_threshold) < 0.0:
            raise ValueError("foot_contact_normal_impulse_threshold must be non-negative")
        if float(self.config.max_abs_root_position) < 0.0:
            raise ValueError("max_abs_root_position must be non-negative")
        if float(self.config.max_abs_root_linear_velocity) < 0.0:
            raise ValueError("max_abs_root_linear_velocity must be non-negative")
        if float(self.config.max_abs_root_angular_velocity) < 0.0:
            raise ValueError("max_abs_root_angular_velocity must be non-negative")
        if float(self.config.max_abs_joint_position) < 0.0:
            raise ValueError("max_abs_joint_position must be non-negative")
        if float(self.config.max_abs_joint_velocity) < 0.0:
            raise ValueError("max_abs_joint_velocity must be non-negative")
        if float(self.config.sparse_target_success_max_base_height) < float(
            self.config.sparse_target_success_min_base_height
        ):
            raise ValueError("sparse_target_success_max_base_height must be >= sparse_target_success_min_base_height")
        if str(self.config.contact_geometry) not in _NANOG1_CONTACT_GEOMETRIES:
            raise ValueError(f"contact_geometry must be one of {_NANOG1_CONTACT_GEOMETRIES}")
        if float(self.config.ground_friction) < 0.0:
            raise ValueError("ground_friction must be non-negative")
        if float(self.config.foot_box_xy_scale) <= 0.0:
            raise ValueError("foot_box_xy_scale must be positive")
        self._reward_mode_id = _g1_reward_mode_id(self.config.reward_mode)
        self._observation_mode_id = _g1_observation_mode_id(self.config.observation_mode)
        self.obs_dim = _g1_observation_dim(self._observation_mode_id)

        self.model = self._build_model()
        self.coord_stride = int(self.model.joint_coord_count) // self.world_count
        self.dof_stride = int(self.model.joint_dof_count) // self.world_count
        if self.coord_stride != 7 + ACTION_DIM_G1 or self.dof_stride != 6 + ACTION_DIM_G1:
            raise RuntimeError(
                f"Expected nanoG1 dimensions coord={7 + ACTION_DIM_G1}, dof={6 + ACTION_DIM_G1}; "
                f"got coord={self.coord_stride}, dof={self.dof_stride}"
            )
        if int(self.model.body_count) % self.world_count != 0:
            raise RuntimeError("Expected equal G1 body count per world")
        self.body_stride = int(self.model.body_count) // self.world_count
        self._left_foot_body_local = self._resolve_foot_body_local("left")
        self._right_foot_body_local = self._resolve_foot_body_local("right")

        self.solver = self._make_solver()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.foot_contacts = wp.zeros((self.world_count, 2), dtype=wp.float32, device=self.device)
        self.foot_air_time = wp.zeros((self.world_count, 2), dtype=wp.float32, device=self.device)
        self.foot_contact_time = wp.zeros((self.world_count, 2), dtype=wp.float32, device=self.device)
        self.foot_timer_episode_step = wp.array(
            np.full(self.world_count, -1, dtype=np.int32), dtype=wp.int32, device=self.device
        )
        self.previous_joint_qd = wp.zeros((self.world_count, self.action_dim), dtype=wp.float32, device=self.device)
        self.joint_acc_l2 = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.joint_regularizer_episode_step = wp.array(
            np.full(self.world_count, -1, dtype=np.int32), dtype=wp.int32, device=self.device
        )
        self._can_scan_foot_contacts = self.model.shape_body is not None and self.model.shape_world is not None

        self.default_joint_pos = wp.array(
            np.asarray(_DEFAULT_JOINT_POS_G1, dtype=np.float32), dtype=wp.float32, device=self.device
        )
        self.ctrl_lower = wp.array(np.asarray(_CTRL_LO_G1, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.ctrl_upper = wp.array(np.asarray(_CTRL_HI_G1, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.actuator_ke = wp.array(np.asarray(_UNITREE_KP_G1, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.actuator_kd = wp.array(np.asarray(_DRIVE_KD_G1, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.actuator_force_kp = wp.array(
            np.asarray(_NANOG1_ACTUATOR_FORCE_KP_G1, dtype=np.float32), dtype=wp.float32, device=self.device
        )
        self.actuator_force_kd = wp.array(
            np.asarray(_NANOG1_ACTUATOR_FORCE_KD_G1, dtype=np.float32), dtype=wp.float32, device=self.device
        )
        self.passive_damping = wp.array(
            np.asarray(_NANOG1_DOF_DAMPING_G1, dtype=np.float32), dtype=wp.float32, device=self.device
        )
        self.actuator_force_lower = wp.array(
            np.asarray(_NANOG1_ACTUATOR_FORCE_LO_G1, dtype=np.float32), dtype=wp.float32, device=self.device
        )
        self.actuator_force_upper = wp.array(
            np.asarray(_NANOG1_ACTUATOR_FORCE_HI_G1, dtype=np.float32), dtype=wp.float32, device=self.device
        )
        self.current_actions = wp.zeros((self.world_count, self.action_dim), dtype=wp.float32, device=self.device)
        self.previous_actions = wp.zeros((self.world_count, self.action_dim), dtype=wp.float32, device=self.device)
        self.actuator_force = wp.zeros((self.world_count, self.action_dim), dtype=wp.float32, device=self.device)
        command_np = np.tile(np.asarray(self.config.command, dtype=np.float32), (self.world_count, 1))
        target_np = np.tile(np.asarray(self.config.sparse_target_position, dtype=np.float32), (self.world_count, 1))
        target_distance_np = np.asarray([np.linalg.norm(target_np[0])], dtype=np.float32)
        self.command = wp.array(command_np, dtype=wp.float32, device=self.device)
        self.target_position = wp.array(target_np, dtype=wp.float32, device=self.device)
        self.target_distance = wp.array(target_distance_np, dtype=wp.float32, device=self.device)
        self.command_scale = wp.array(np.ones(1, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.episode_steps = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)
        self.obs = wp.zeros((self.world_count, self.obs_dim), dtype=wp.float32, device=self.device)
        self.rewards = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.dones = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self._reset_articulation_mask = wp.zeros(self.world_count, dtype=wp.bool, device=self.device)
        self.successes = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.step_rewards = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.step_dones = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.step_successes = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self._reset_seed = 0
        self._reset_seed_counter: wp.array[wp.int32] | None = None
        self._command_seed = 0
        self._command_seed_counter: wp.array[wp.int32] | None = None
        self.sim_time = 0.0

        self.reset()

    def _build_model(self):
        articulation_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(articulation_builder)
        articulation_builder.default_shape_cfg.ke = 5.0e4
        articulation_builder.default_shape_cfg.kd = 5.0e2
        articulation_builder.default_shape_cfg.kf = 1.0e3
        articulation_builder.default_shape_cfg.mu = 0.75
        articulation_builder.default_shape_cfg.gap = 0.0

        asset_path = newton.utils.download_asset("unitree_g1")
        mjcf_path = asset_path / "mjcf" / "g1_29dof.xml"
        parse_visuals = bool(self.config.parse_visuals)
        parse_meshes = bool(self.config.parse_meshes)
        mjcf_source = str(mjcf_path)
        if parse_visuals and not parse_meshes:
            mjcf_source = _g1_mjcf_with_visual_meshes_without_mesh_colliders(mjcf_path)
            parse_meshes = True
        articulation_builder.add_mjcf(
            mjcf_source,
            floating=None,
            enable_self_collisions=False,
            ignore_names=("floor", "ground"),
            parse_visuals=parse_visuals,
            parse_meshes=parse_meshes,
            ignore_inertial_definitions=False,
        )
        self._apply_contact_geometry(articulation_builder)
        _apply_nanog1_body_inertials(articulation_builder)
        articulation_builder.joint_q[:3] = [0.0, 0.0, 0.785]
        articulation_builder.joint_q[3:7] = [0.0, 0.0, 0.0, 1.0]
        articulation_builder.joint_q[7 : 7 + ACTION_DIM_G1] = list(_DEFAULT_JOINT_POS_G1)
        actuator_target_mode = (
            int(newton.JointTargetMode.EFFORT)
            if str(self.config.actuation_model) == "explicit_torque"
            else int(newton.JointTargetMode.POSITION)
        )
        for i in range(ACTION_DIM_G1):
            dof = i + 6
            articulation_builder.joint_target_ke[dof] = _UNITREE_KP_G1[i]
            articulation_builder.joint_target_kd[dof] = _DRIVE_KD_G1[i]
            articulation_builder.joint_limit_lower[dof] = _CTRL_LO_G1[i]
            articulation_builder.joint_limit_upper[dof] = _CTRL_HI_G1[i]
            articulation_builder.joint_effort_limit[dof] = _NANOG1_ACTUATOR_FORCE_HI_G1[i]
            articulation_builder.joint_damping[dof] = _NANOG1_DOF_DAMPING_G1[i]
            articulation_builder.joint_friction[dof] = (
                float(self.config.joint_friction_scale) * _NANOG1_DOF_FRICTIONLOSS_G1[i]
            )
            articulation_builder.joint_armature[dof] = _NANOG1_DOF_ARMATURE_G1[i]
            articulation_builder.joint_target_mode[dof] = actuator_target_mode

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(self.world_count):
            builder.add_world(articulation_builder)
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = float(self.config.ground_friction)
        builder.default_shape_cfg.gap = 0.0
        builder.add_ground_plane()
        model = builder.finalize(device=self.device)
        model.set_gravity((0.0, 0.0, -9.81))
        contact_cap_per_world = int(self.config.rigid_contact_max_per_world)
        if contact_cap_per_world > 0 and not bool(self.config.parse_meshes):
            model._collision_pipeline = newton.CollisionPipeline(
                model,
                contact_matching=PHOENX_CONTACT_MATCHING,
                rigid_contact_max=max(1, self.world_count * contact_cap_per_world),
            )
        return model

    def _apply_contact_geometry(self, builder: newton.ModelBuilder) -> None:
        contact_geometry = str(self.config.contact_geometry)
        if contact_geometry == _NANOG1_CONTACT_GEOMETRY_MJCF:
            return
        if contact_geometry == _NANOG1_CONTACT_GEOMETRY_FOOT_BOXES:
            self._apply_nanog1_foot_boxes(builder)
            return
        raise ValueError(f"contact_geometry must be one of {_NANOG1_CONTACT_GEOMETRIES}")

    def _apply_nanog1_foot_boxes(self, builder: newton.ModelBuilder) -> None:
        left_body = self._disable_ankle_point_contacts(builder, "left")
        right_body = self._disable_ankle_point_contacts(builder, "right")
        cfg = builder.default_shape_cfg.copy()
        cfg.mu = _NANOG1_FOOT_BOX_MU
        cfg.gap = 0.0
        if bool(self.config.parse_visuals):
            cfg.is_visible = False
        hx, hy, hz = _NANOG1_FOOT_BOX_HALF_EXTENTS
        xy_scale = float(self.config.foot_box_xy_scale)
        hx *= xy_scale
        hy *= xy_scale
        xform = wp.transform(p=wp.vec3(*_NANOG1_FOOT_BOX_LOCAL_POS), q=wp.quat_identity())
        imported_shape_count = len(builder.shape_type)
        left_shape = builder.add_shape_box(
            body=left_body,
            xform=xform,
            hx=hx,
            hy=hy,
            hz=hz,
            cfg=cfg,
            label="left_nanog1_foot_box",
        )
        right_shape = builder.add_shape_box(
            body=right_body,
            xform=xform,
            hx=hx,
            hy=hy,
            hz=hz,
            cfg=cfg,
            label="right_nanog1_foot_box",
        )
        for foot_shape in (left_shape, right_shape):
            for shape in range(imported_shape_count):
                builder.add_shape_collision_filter_pair(foot_shape, shape)
        builder.add_shape_collision_filter_pair(left_shape, right_shape)

    @staticmethod
    def _disable_ankle_point_contacts(builder: newton.ModelBuilder, side: str) -> int:
        body_ids: list[int] = []
        collide_bit = int(newton.ShapeFlags.COLLIDE_SHAPES)
        particle_bit = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        geom_token = f"{side}_ankle_roll_link_geom_"
        for shape_index, label in enumerate(builder.shape_label):
            if geom_token not in label:
                continue
            body_ids.append(int(builder.shape_body[shape_index]))
            builder.shape_flags[shape_index] = int(builder.shape_flags[shape_index]) & ~collide_bit & ~particle_bit
        unique_body_ids = sorted(set(body_ids))
        if len(unique_body_ids) != 1:
            raise RuntimeError(f"Expected one {side} ankle contact body, found {unique_body_ids}")
        return unique_body_ids[0]

    def _make_solver(self):
        # The environment loop already runs sim_substeps solver calls per policy
        # step; nesting SolverPhoenX substeps would square the recipe value.
        return newton.solvers.SolverPhoenX(
            self.model,
            substeps=1,
            solver_iterations=int(self.config.solver_iterations),
            velocity_iterations=int(self.config.velocity_iterations),
            joint_friction_model=str(self.config.joint_friction_model),
            threads_per_world=self.config.threads_per_world,
            multi_world_scheduler=self.config.multi_world_scheduler,
            prepare_refresh_stride=self.config.prepare_refresh_stride,
            articulation_mode=str(self.config.articulation_mode),
        )

    def _resolve_foot_body_local(self, side: str) -> int:
        shape_body = self.model.shape_body.numpy() if self.model.shape_body is not None else None
        shape_label = f"{side}_nanog1_foot_box"
        if shape_body is not None:
            for shape_index, label in enumerate(getattr(self.model, "shape_label", ())):
                if label == shape_label:
                    body = int(shape_body[shape_index])
                    if body >= 0:
                        return body % self.body_stride

        body_suffix = f"{side}_ankle_roll_link"
        for body_index, label in enumerate(getattr(self.model, "body_label", ())):
            if str(label).endswith(body_suffix):
                return int(body_index) % self.body_stride
        return -1

    def set_command(self, command: tuple[float, float, float]) -> None:
        """Set the same body-frame command for every world [m/s, m/s, rad/s]."""

        cmd = (float(command[0]), float(command[1]), float(command[2]))
        self.config.command = cmd
        command_np = np.tile(np.asarray(cmd, dtype=np.float32), (self.world_count, 1))
        self.command.assign(command_np)

    def set_commands(self, commands: np.ndarray) -> None:
        """Set per-world body-frame commands [m/s, m/s, rad/s]."""

        cmds = np.asarray(commands, dtype=np.float32)
        if cmds.shape != (self.world_count, 3):
            raise ValueError(f"Expected commands with shape {(self.world_count, 3)}, got {cmds.shape}")
        self.config.command = (float(cmds[0, 0]), float(cmds[0, 1]), float(cmds[0, 2]))
        self.command.assign(cmds)

    def set_target_position(self, target_position: tuple[float, float]) -> None:
        """Set the same sparse target XY position for every world [m]."""

        target = (float(target_position[0]), float(target_position[1]))
        self.config.sparse_target_position = target
        target_np = np.tile(np.asarray(target, dtype=np.float32), (self.world_count, 1))
        self.target_position.assign(target_np)
        self.target_distance.assign(np.asarray([np.linalg.norm(target_np[0])], dtype=np.float32))

    def set_target_positions(self, target_positions: np.ndarray) -> None:
        """Set per-world sparse target XY positions [m]."""

        targets = np.asarray(target_positions, dtype=np.float32)
        if targets.shape != (self.world_count, 2):
            raise ValueError(f"Expected target positions with shape {(self.world_count, 2)}, got {targets.shape}")
        self.config.sparse_target_position = (float(targets[0, 0]), float(targets[0, 1]))
        self.target_position.assign(targets)
        self.target_distance.assign(np.asarray([np.linalg.norm(targets[0])], dtype=np.float32))

    def update_sparse_target_distance_curriculum(
        self,
        sample_counter: wp.array[wp.int32],
        *,
        sample_delta: int,
        start_distance: float,
        end_distance: float,
        ramp_samples: float,
    ) -> None:
        """Advance the device-side sparse-target distance curriculum [m]."""

        wp.launch(
            g1_update_sparse_target_distance_kernel,
            dim=1,
            inputs=[
                sample_counter,
                int(sample_delta),
                float(start_distance),
                float(end_distance),
                float(ramp_samples),
            ],
            outputs=[self.target_distance],
            device=self.device,
        )

    def set_sparse_targets_from_distance(self) -> None:
        """Set all sparse targets from the current device-side target distance."""

        base = self.config.sparse_target_position
        wp.launch(
            g1_set_sparse_targets_from_distance_kernel,
            dim=(self.world_count, 2),
            inputs=[float(base[0]), float(base[1]), self.target_distance],
            outputs=[self.target_position],
            device=self.device,
        )

    def randomize_target_positions_seed_counter(
        self,
        *,
        seed_counter: wp.array[wp.int32],
        target_angle_range: tuple[float, float],
        seed_offset: int = 0,
        advance: int = 1,
        target_distance_min: float | None = None,
    ) -> None:
        """Sample per-world sparse targets using a graph-replay-safe device seed counter."""

        angle_min = float(target_angle_range[0])
        angle_max = float(target_angle_range[1])
        if angle_max < angle_min:
            raise ValueError("target_angle_range must be ordered")
        base = self.config.sparse_target_position
        base_angle = float(np.arctan2(float(base[1]), float(base[0])))
        if target_distance_min is None:
            wp.launch(
                g1_sample_sparse_targets_seed_counter_kernel,
                dim=(self.world_count, 2),
                inputs=[seed_counter, int(seed_offset), base_angle, angle_min, angle_max, self.target_distance],
                outputs=[self.target_position],
                device=self.device,
            )
        else:
            wp.launch(
                g1_sample_sparse_target_range_seed_counter_kernel,
                dim=(self.world_count, 2),
                inputs=[
                    seed_counter,
                    int(seed_offset),
                    base_angle,
                    angle_min,
                    angle_max,
                    float(target_distance_min),
                    self.target_distance,
                ],
                outputs=[self.target_position],
                device=self.device,
            )
        if int(advance) != 0:
            advance_seed_counter(seed_counter, int(advance), device=self.device)

    def randomize_commands(
        self,
        *,
        seed: int,
        command_x_range: tuple[float, float],
        command_y_range: tuple[float, float],
        command_yaw_range: tuple[float, float],
        zero_probability: float = 0.0,
    ) -> None:
        """Sample per-world body-frame commands on the device [m/s, m/s, rad/s]."""

        x_min, x_max, y_min, y_max, yaw_min, yaw_max = self._validate_command_ranges(
            command_x_range, command_y_range, command_yaw_range
        )
        wp.launch(
            g1_sample_commands_kernel,
            dim=(self.world_count, 3),
            inputs=[
                int(seed),
                x_min,
                x_max,
                y_min,
                y_max,
                yaw_min,
                yaw_max,
                self.command_scale,
                float(zero_probability),
            ],
            outputs=[self.command],
            device=self.device,
        )

    def randomize_commands_seed_counter(
        self,
        *,
        seed_counter: wp.array[wp.int32],
        command_x_range: tuple[float, float],
        command_y_range: tuple[float, float],
        command_yaw_range: tuple[float, float],
        zero_probability: float = 0.0,
        seed_offset: int = 0,
        advance: int = 1,
    ) -> None:
        """Sample per-world commands using a graph-replay-safe device seed counter."""

        x_min, x_max, y_min, y_max, yaw_min, yaw_max = self._validate_command_ranges(
            command_x_range, command_y_range, command_yaw_range
        )
        wp.launch(
            g1_sample_commands_seed_counter_kernel,
            dim=(self.world_count, 3),
            inputs=[
                seed_counter,
                int(seed_offset),
                x_min,
                x_max,
                y_min,
                y_max,
                yaw_min,
                yaw_max,
                self.command_scale,
                float(zero_probability),
            ],
            outputs=[self.command],
            device=self.device,
        )
        if int(advance) != 0:
            advance_seed_counter(seed_counter, int(advance), device=self.device)

    @staticmethod
    def _check_command_ranges(
        command_x_range: tuple[float, float],
        command_y_range: tuple[float, float],
        command_yaw_range: tuple[float, float],
    ) -> tuple[float, float, float, float, float, float]:
        x_min, x_max = float(command_x_range[0]), float(command_x_range[1])
        y_min, y_max = float(command_y_range[0]), float(command_y_range[1])
        yaw_min, yaw_max = float(command_yaw_range[0]), float(command_yaw_range[1])
        if x_max < x_min or y_max < y_min or yaw_max < yaw_min:
            raise ValueError("command ranges must be ordered")
        return x_min, x_max, y_min, y_max, yaw_min, yaw_max

    def _validate_command_ranges(
        self,
        command_x_range: tuple[float, float],
        command_y_range: tuple[float, float],
        command_yaw_range: tuple[float, float],
    ) -> tuple[float, float, float, float, float, float]:
        x_min, x_max, y_min, y_max, yaw_min, yaw_max = self._check_command_ranges(
            command_x_range, command_y_range, command_yaw_range
        )
        self.config.command = ((x_min + x_max) * 0.5, (y_min + y_max) * 0.5, (yaw_min + yaw_max) * 0.5)
        return x_min, x_max, y_min, y_max, yaw_min, yaw_max

    def _config_command_ranges(self) -> tuple[float, float, float, float, float, float]:
        return self._check_command_ranges(
            self.config.command_x_range, self.config.command_y_range, self.config.command_yaw_range
        )

    def observe(self) -> wp.array:
        """Update and return the current observation array."""

        self.foot_contacts.zero_()
        contact_container = getattr(self.solver.world, "_contact_container", None)
        if self._can_scan_foot_contacts and contact_container is not None and int(self.contacts.rigid_contact_max) > 0:
            wp.launch(
                g1_scan_foot_contacts_kernel,
                dim=int(self.contacts.rigid_contact_max),
                inputs=[
                    self.contacts.rigid_contact_count,
                    self.contacts.rigid_contact_shape0,
                    self.contacts.rigid_contact_shape1,
                    self.model.shape_body,
                    self.model.shape_world,
                    contact_container,
                    self.body_stride,
                    self._left_foot_body_local,
                    self._right_foot_body_local,
                    self.config.foot_contact_normal_impulse_threshold,
                ],
                outputs=[self.foot_contacts],
                device=self.device,
            )

        wp.launch(
            g1_observe_reward_kernel,
            dim=(self.world_count, self.obs_dim),
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.default_joint_pos,
                self.current_actions,
                self.previous_actions,
                self.command,
                self.target_position,
                self.state_0.body_q,
                self.state_0.body_qd,
                self.model.body_com,
                self.foot_contacts,
                self.actuator_force,
                self.coord_stride,
                self.dof_stride,
                self.body_stride,
                self._left_foot_body_local,
                self._right_foot_body_local,
                self.episode_steps,
                self.config.max_episode_steps,
                self.config.phase_period,
                self.config.min_base_height,
                self.config.min_upright_cos,
                self.config.max_abs_root_position,
                self.config.max_abs_root_linear_velocity,
                self.config.max_abs_root_angular_velocity,
                self.config.max_abs_joint_position,
                self.config.max_abs_joint_velocity,
                self.config.action_scale,
                self.config.frame_dt,
                self.config.w_track_lin,
                self.config.w_track_ang,
                self.config.w_command_progress,
                self.config.w_lin_vel_z,
                self.config.w_ang_vel_xy,
                self.config.w_orientation,
                self.config.w_torque,
                self.config.w_action_rate,
                self.config.w_alive,
                self.config.w_termination,
                self.config.gait_stance_fraction,
                self.config.w_gait_contact,
                self.config.w_gait_swing,
                self.config.w_gait_swing_contact,
                self.config.w_gait_hip,
                self.config.gait_foot_height,
                self.config.w_base_height,
                self.config.base_height_target,
                self.config.w_feet_air_time,
                self.config.feet_air_time_threshold,
                self.config.w_feet_slide,
                self.ctrl_lower,
                self.ctrl_upper,
                self.config.w_joint_deviation_hip,
                self.config.w_joint_deviation_waist,
                self.config.w_joint_deviation_upper,
                self.config.w_joint_acc_legs,
                self.config.w_joint_pos_limit_ankle,
                self._reward_mode_id,
                self._observation_mode_id,
                self.config.w_sparse_command_success,
                self.config.w_target_progress,
                self.config.sparse_command_velocity_tolerance,
                self.config.sparse_command_yaw_tolerance,
                self.config.sparse_target_radius,
                self.config.sparse_target_success_upright_cos,
                self.config.sparse_target_success_min_base_height,
                self.config.sparse_target_success_max_base_height,
                self.config.w_mechanical_power,
            ],
            outputs=[
                self.foot_air_time,
                self.foot_contact_time,
                self.foot_timer_episode_step,
                self.previous_joint_qd,
                self.joint_acc_l2,
                self.joint_regularizer_episode_step,
                self.obs,
                self.rewards,
                self.dones,
                self.successes,
            ],
            device=self.device,
        )
        return self.obs

    def _clear_reset_solver_state(self) -> None:
        world = self.solver.world
        actuated_double_ball_socket_clear_reset_worlds(
            world.constraints,
            world.bodies,
            world.num_joints,
            self.dones,
            device=self.device,
        )
        constraint_container_clear_reset_worlds(
            world.constraints,
            world._contact_offset,
            world.num_bodies,
            world.bodies.world_id,
            getattr(world, "_particle_world_id", None),
            self.dones,
            device=self.device,
        )
        shape_world = getattr(self.model, "shape_world", None)
        cid_of_contact_cur = getattr(world, "_cid_of_contact_cur", None)
        cid_of_contact_prev = getattr(world, "_cid_of_contact_prev", None)
        if shape_world is None or cid_of_contact_cur is None or cid_of_contact_prev is None:
            return
        contact_container_clear_reset_worlds(
            world._contact_container,
            cid_of_contact_cur,
            cid_of_contact_prev,
            self.contacts,
            shape_world,
            self.dones,
            device=self.device,
        )

    def use_command_seed_counter(self, seed_counter: wp.array[wp.int32] | None) -> None:
        """Use a device seed counter for graph-captured command sampling."""

        self._command_seed_counter = seed_counter

    def update_command_curriculum(
        self,
        sample_counter: wp.array[wp.int32],
        *,
        sample_delta: int,
        start_scale: float,
        ramp_samples: float,
    ) -> None:
        """Advance the device-side command curriculum scale."""

        wp.launch(
            g1_update_command_scale_kernel,
            dim=1,
            inputs=[sample_counter, int(sample_delta), float(start_scale), float(ramp_samples)],
            outputs=[self.command_scale],
            device=self.device,
        )

    def _sample_done_commands(self) -> bool:
        if not bool(self.config.randomize_commands_on_reset):
            return False
        x_min, x_max, y_min, y_max, yaw_min, yaw_max = self._config_command_ranges()
        zero_probability = float(self.config.command_zero_probability)
        if self._command_seed_counter is None:
            wp.launch(
                g1_sample_done_commands_kernel,
                dim=(self.world_count, 3),
                inputs=[
                    int(self._command_seed),
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    yaw_min,
                    yaw_max,
                    self.command_scale,
                    zero_probability,
                    self.dones,
                ],
                outputs=[self.command],
                device=self.device,
            )
            self._command_seed += 1
        else:
            wp.launch(
                g1_sample_done_commands_seed_counter_kernel,
                dim=(self.world_count, 3),
                inputs=[
                    self._command_seed_counter,
                    0,
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    yaw_min,
                    yaw_max,
                    self.command_scale,
                    zero_probability,
                    self.dones,
                ],
                outputs=[self.command],
                device=self.device,
            )
            advance_seed_counter(self._command_seed_counter, 1, device=self.device)
        return True

    def _resample_due_commands(self) -> bool:
        if not bool(self.config.randomize_commands_on_reset):
            return False
        resample_steps = int(self.config.command_resample_steps)
        if resample_steps <= 0:
            return False
        x_min, x_max, y_min, y_max, yaw_min, yaw_max = self._config_command_ranges()
        zero_probability = float(self.config.command_zero_probability)
        if self._command_seed_counter is None:
            wp.launch(
                g1_sample_due_commands_kernel,
                dim=(self.world_count, 3),
                inputs=[
                    int(self._command_seed),
                    resample_steps,
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    yaw_min,
                    yaw_max,
                    self.command_scale,
                    zero_probability,
                    self.dones,
                    self.episode_steps,
                ],
                outputs=[self.command],
                device=self.device,
            )
            self._command_seed += 1
        else:
            wp.launch(
                g1_sample_due_commands_seed_counter_kernel,
                dim=(self.world_count, 3),
                inputs=[
                    self._command_seed_counter,
                    0,
                    resample_steps,
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    yaw_min,
                    yaw_max,
                    self.command_scale,
                    zero_probability,
                    self.dones,
                    self.episode_steps,
                ],
                outputs=[self.command],
                device=self.device,
            )
            advance_seed_counter(self._command_seed_counter, 1, device=self.device)
        return True

    def reset(self) -> wp.array:
        """Reset all worlds and return observations."""

        wp.copy(self.state_0.joint_q, self.model.joint_q)
        wp.copy(self.state_0.joint_qd, self.model.joint_qd)
        self.current_actions.zero_()
        self.previous_actions.zero_()
        self.actuator_force.zero_()
        self.foot_contacts.zero_()
        self.foot_air_time.zero_()
        self.foot_contact_time.zero_()
        self.foot_timer_episode_step.fill_(-1)
        self.previous_joint_qd.zero_()
        self.joint_acc_l2.zero_()
        self.joint_regularizer_episode_step.fill_(-1)
        self.control.joint_f.zero_()
        self.episode_steps.zero_()
        self.dones.fill_(1.0)
        self._clear_reset_solver_state()
        self._sample_done_commands()
        self.dones.zero_()
        self.successes.zero_()
        self.step_rewards.zero_()
        self.step_dones.zero_()
        self.step_successes.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.sim_time = 0.0
        return self.observe()

    def reset_noisy(self, seed: int | None = None) -> wp.array:
        """Reset all worlds with configured joint-position reset noise."""

        self.reset()
        if seed is not None:
            self._reset_seed = int(seed)
        self.dones.assign(np.ones(self.world_count, dtype=np.float32))
        self.reset_done()
        self.dones.zero_()
        return self.observe()

    def reset_done(self) -> None:
        """Reset worlds whose done flag is set."""

        max_cols = max(self.coord_stride, self.dof_stride, self.action_dim)
        wp.launch(
            g1_reset_done_worlds_kernel,
            dim=(self.world_count, max_cols),
            inputs=[
                int(self._reset_seed),
                self.config.reset_noise,
                self.dones,
                self.model.joint_q,
                self.model.joint_qd,
                self.coord_stride,
                self.dof_stride,
                self.action_dim,
            ],
            outputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.episode_steps,
                self.previous_actions,
                self.current_actions,
                self.actuator_force,
                self.previous_joint_qd,
                self.joint_acc_l2,
                self.joint_regularizer_episode_step,
                self.foot_air_time,
                self.foot_contact_time,
                self.foot_timer_episode_step,
                self.control.joint_f,
                self._reset_articulation_mask,
            ],
            device=self.device,
        )
        self._reset_seed += 1
        self._clear_reset_solver_state()
        self._sample_done_commands()
        newton.eval_fk(
            self.model,
            self.state_0.joint_q,
            self.state_0.joint_qd,
            self.state_0,
            mask=self._reset_articulation_mask,
        )

    def use_reset_seed_counter(self, seed_counter: wp.array[wp.int32] | None) -> None:
        """Use a device seed counter for graph-captured reset noise."""

        self._reset_seed_counter = seed_counter

    def reset_done_seed_counter(
        self, seed_counter: wp.array[wp.int32], *, seed_offset: int = 0, advance: int = 1
    ) -> None:
        """Reset done worlds using a graph-replay-safe device seed counter."""

        max_cols = max(self.coord_stride, self.dof_stride, self.action_dim)
        wp.launch(
            g1_reset_done_worlds_seed_counter_kernel,
            dim=(self.world_count, max_cols),
            inputs=[
                seed_counter,
                int(seed_offset),
                self.config.reset_noise,
                self.dones,
                self.model.joint_q,
                self.model.joint_qd,
                self.coord_stride,
                self.dof_stride,
                self.action_dim,
            ],
            outputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.episode_steps,
                self.previous_actions,
                self.current_actions,
                self.actuator_force,
                self.previous_joint_qd,
                self.joint_acc_l2,
                self.joint_regularizer_episode_step,
                self.foot_air_time,
                self.foot_contact_time,
                self.foot_timer_episode_step,
                self.control.joint_f,
                self._reset_articulation_mask,
            ],
            device=self.device,
        )
        if int(advance) != 0:
            advance_seed_counter(seed_counter, int(advance), device=self.device)
        self._clear_reset_solver_state()
        self._sample_done_commands()
        newton.eval_fk(
            self.model,
            self.state_0.joint_q,
            self.state_0.joint_qd,
            self.state_0,
            mask=self._reset_articulation_mask,
        )

    def _gather_actuator_force(self, *, scatter_joint_f: bool = False) -> None:
        wp.launch(
            g1_gather_actuator_force_kernel,
            dim=(self.world_count, self.action_dim),
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.control.joint_target_q,
                self.actuator_force_kp,
                self.actuator_force_kd,
                self.passive_damping,
                self.actuator_force_lower,
                self.actuator_force_upper,
                self.coord_stride,
                self.dof_stride,
                int(bool(self.model.use_coord_layout_targets)),
                int(bool(scatter_joint_f)),
            ],
            outputs=[self.actuator_force, self.control.joint_f],
            device=self.device,
        )

    def step(self, actions: wp.array) -> tuple[wp.array, wp.array, wp.array]:
        """Apply actions, advance PhoenX, and return ``(obs, rewards, dones)``."""

        wp.launch(
            g1_apply_actions_kernel,
            dim=(self.world_count, self.action_dim),
            inputs=[
                actions,
                self.default_joint_pos,
                self.ctrl_lower,
                self.ctrl_upper,
                self.config.action_scale,
                int(self.config.controlled_action_count),
                self.dof_stride,
                self.coord_stride,
                int(bool(self.model.use_coord_layout_targets)),
            ],
            outputs=[self.current_actions, self.control.joint_target_q],
            device=self.device,
        )

        substeps = int(self.config.sim_substeps)
        sub_dt = float(self.config.frame_dt) / float(substeps)
        explicit_torque = str(self.config.actuation_model) == "explicit_torque"
        for substep in range(substeps):
            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            if explicit_torque or substep == substeps - 1:
                self._gather_actuator_force(scatter_joint_f=explicit_torque)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        wp.launch(
            g1_increment_episode_steps_kernel,
            dim=self.world_count,
            outputs=[self.episode_steps],
            device=self.device,
        )
        self.observe()
        wp.copy(self.step_rewards, self.rewards)
        wp.copy(self.step_dones, self.dones)
        wp.copy(self.step_successes, self.successes)
        wp.copy(self.previous_actions, self.current_actions)
        update_obs = False
        if self.config.auto_reset:
            if self._reset_seed_counter is None:
                self.reset_done()
            else:
                self.reset_done_seed_counter(self._reset_seed_counter)
            update_obs = True
        if self._resample_due_commands():
            update_obs = True
        if update_obs:
            self.observe()
        self.sim_time += float(self.config.frame_dt)
        return self.obs, self.step_rewards, self.step_dones

    def collect_ppo_rollout(
        self,
        trainer: TrainerPPO,
        buffer: BufferRollout,
        *,
        seed: int,
        reset_state_at_start: bool = True,
    ) -> None:
        """Collect one rollout and compute GAE returns for PPO."""

        collect_ppo_rollout(self, trainer, buffer, seed=seed, reset_state_at_start=reset_state_at_start)

    def collect_ppo_rollout_seed_counter(
        self,
        trainer: TrainerPPO,
        buffer: BufferRollout,
        *,
        seed_counter: wp.array[wp.int32],
        reset_state_at_start: bool = True,
    ) -> None:
        """Collect one rollout using a graph-replay-safe device seed counter."""

        collect_ppo_rollout_seed_counter(
            self, trainer, buffer, seed_counter=seed_counter, reset_state_at_start=reset_state_at_start
        )
