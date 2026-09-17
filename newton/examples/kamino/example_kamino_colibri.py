# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Colibri mechanical hummingbird, built directly from meter-based OBJ assets.

Command: python -m newton.examples kamino_colibri
Use --body-count N to assemble and test progressively from FrameGround.
All body, shape, and joint data below were extracted once from Colibri.usd.
The example does not load or require USD.
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import trimesh
import warp as wp

import newton
import newton.examples

BODY_ORDER = [
    "FrameGround",
    "Frame",
    "Crank",
    "HummerBody",
    "GearedSpinner",
    "CamFollowerBody",
    "CamWheelHead",
    "Gear_Large__3x_02",
    "Gear_Large__3x_01",
    "Gear_Large__3x_00",
    "CamWheelBottom",
    "CamWheelTail",
    "CamFollower",
    "CamFollowerHead",
    "Hypocycloid_Gear__3x_0",
    "Hypocycloid_Gear__3x_02",
    "Hypocycloid_Gear__3x_01",
    "HummerHead",
    "WingLinkArcLeft",
    "WingLinkArcRight",
    "WingLinkStraightRight",
    "WingLinkStraightLeft",
    "ShoulderRight",
    "ShoulderLeft",
    "TailMount",
    "Tail_Feather_A__2x_",
    "Tail_Feather_B__2x_",
    "Tail_Feather_C",
    "Tail_Feather_A__2x__mirrored",
    "Tail_Feather_B__2x__mirrored",
    "WingRight",
    "WingLeft",
    "TailPinion",
    "WingRightConnector",
    "WingLeftConnector",
    "TailRack",
]

BODY_POSES = {
    "Hypocycloid_Gear__3x_0": (
        -0.037798548,
        0.025717562,
        0.154258906,
        0.704159916,
        0.068311171,
        0.068920497,
        -0.703379245,
    ),
    "Hypocycloid_Gear__3x_02": (
        -0.012099686,
        0.025574885,
        0.025940897,
        0.707219601,
        0.035365787,
        0.035358817,
        -0.70522298,
    ),
    "Hypocycloid_Gear__3x_01": (
        -0.1040044,
        0.025774127,
        0.119656086,
        -0.665990598,
        0.237631997,
        0.237402776,
        0.666053661,
    ),
    "Crank": (0.098308258, 0.025566282, -0.124117225, -0.685761749, 0.167254073, 0.165750133, 0.688682649),
    "HummerBody": (0.052513877, 0.024970772, 0.236781483, -0.008622916, 0.707109975, 0.706994462, 0.008941949),
    "GearedSpinner": (0.052512161, 0.024973803, 0.2367836, 0.654883973, 0.266715324, 0.266941524, -0.654776405),
    "CamFollowerBody": (0.085862178, 0.024961496, 0.202840118, 0.296674611, 0.641868519, 0.641957566, -0.296444705),
    "Tail_Feather_A__2x_": (0.069196341, 0.013328295, 0.259774571, 0.066192409, 0.683049644, 0.712829359, -0.144692963),
    "Tail_Feather_B__2x_": (0.06870722, 0.018768449, 0.259951936, 0.085014731, 0.690895636, 0.706692238, -0.126577237),
    "Tail_Feather_C": (0.06848004, 0.024088592, 0.259977342, 0.103258093, 0.698007432, 0.700196234, -0.108851386),
    "Tail_Feather_A__2x__mirrored": (
        0.065202597,
        0.036566884,
        0.253138181,
        0.13940919,
        0.710663534,
        0.685677337,
        -0.073273519,
    ),
    "Tail_Feather_B__2x__mirrored": (
        0.066677968,
        0.030636399,
        0.256641389,
        0.119220098,
        0.703834383,
        0.694055936,
        -0.093220635,
    ),
    "CamWheelHead": (-0.035263439, 0.025003099, 0.155671433, 0.703913927, 0.067967509, 0.06829395, -0.703719786),
    "Gear_Large__3x_02": (0.04207173, 0.024969034, 0.017203833, 0.628304785, 0.324423405, 0.324626065, -0.628172325),
    "Gear_Large__3x_01": (-0.066952855, 0.025013967, 0.183274039, -0.437513737, 0.555517172, 0.555348909, 0.437687093),
    "Gear_Large__3x_00": (-0.006371005, 0.024992267, 0.146237106, 0.665568589, 0.238825487, 0.239041688, -0.665462179),
    "CamWheelBottom": (0.074067862, 0.024631772, -0.07633222, 0.707026482, 0.031222623, 0.032466712, -0.705751099),
    "CamWheelTail": (-0.153575391, 0.025044227, 0.068887884, -0.659375022, 0.2553983, 0.255159588, 0.659461806),
    "TailRack": (0.056645605, 0.025029334, 0.264481964, 0.103219491, 0.701777709, 0.696852701, -0.106066478),
    "CamFollower": (0.041124167, 0.02497517, 0.254303017, 0.063359548, 0.704263679, 0.704283944, -0.063105983),
    "CamFollowerHead": (0.077591947, 0.024960942, 0.032473914, -0.037496114, 0.706114272, 0.706095974, 0.037751187),
    "HummerHead": (0.049185395, 0.024981048, 0.237519302, 0.022977583, 0.706827206, 0.706652444, -0.022575519),
    "WingLinkArcLeft": (0.081288223, 0.024954953, 0.254676649, -0.116529144, 0.697441082, 0.697348754, 0.117053885),
    "WingLinkArcRight": (0.154836463, 0.024883032, 0.050102578, -0.121237778, 0.696652642, 0.696519307, 0.121808669),
    "WingLinkStraightRight": (
        0.049102063,
        0.024985428,
        0.278357886,
        0.369833094,
        0.602699672,
        0.602977312,
        -0.369316868,
    ),
    "WingLinkStraightLeft": (0.049591635, 0.024978227, 0.278818933, 0.374291761, 0.59992376, 0.600202229, -0.373837458),
    "WingRight": (0.107228338, 0.049153829, 0.231723137, 0.469570041, -0.399603188, -0.42830527, 0.660587514),
    "ShoulderRight": (0.111250226, 0.046505885, 0.238238888, 0.529603218, -0.481508383, -0.468455652, 0.517899034),
    "WingRightConnector": (0.056119655, 0.025984227, 0.255298481, 0.455017586, 0.645023986, 0.589720839, -0.170682119),
    "WingLeft": (0.107936288, -0.000349228, 0.23161358, 0.67272047, 0.417160426, 0.3951484, 0.466135269),
    "ShoulderLeft": (0.112604505, -0.000796232, 0.238277646, 0.541247298, 0.441470928, 0.454968876, 0.552411173),
    "WingLeftConnector": (-0.117687109, 0.016237038, 0.129540541, 0.179982718, 0.559973991, 0.670366853, -0.452375544),
    "TailMount": (0.068116957, 0.024964915, 0.258844948, 0.106245467, 0.699133577, 0.699074329, -0.10592084),
    "TailPinion": (0.099065845, 0.024952912, 0.269358979, 0.315137494, 0.633060598, 0.633104406, -0.314803827),
    "Frame": (0.052507983, 0.024972811, 0.236783142, 0.116273583, 0.697487881, 0.69751661, -0.116024517),
    "FrameGround": (-0.003510761, 0.024991156, 0.215598183, 0.000212737, 0.70708608, 0.707127445, 8.5451e-05),
}

SHAPES = [
    (
        "Hypocycloid_Gear__3x_0",
        "mesh",
        "Hypocycloid_Gear__3x_0/Hypocycloid_Gear__3x_0_mesh",
        "Hypocycloid Gear (3x).obj",
        ((1.0, -0.0, -0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)),
    ),
    (
        "Hypocycloid_Gear__3x_02",
        "mesh",
        "Hypocycloid_Gear__3x_02/Hypocycloid_Gear__3x_0_mesh",
        "Hypocycloid Gear (3x).obj",
        ((1.0, 0.0, -0.0, -0.0), (0.0, 1.0, 0.0, 0.0), (0.0, -0.0, 1.0, 0.0)),
    ),
    (
        "Hypocycloid_Gear__3x_01",
        "mesh",
        "Hypocycloid_Gear__3x_01/Hypocycloid_Gear__3x_0_mesh",
        "Hypocycloid Gear (3x).obj",
        ((1.0000010729, -0.0, -0.0, 0.0), (-0.0, 1.0000010729, -0.0, 0.0), (-0.0, 0.0, 1.0, 0.0)),
    ),
    (
        "Crank",
        "mesh",
        "Crank/Gear_Small_Lower",
        "Gear Small Lower.obj",
        ((1.0, 0.0, 0.0, 0.0), (-0.0, 1.0, 0.0, 0.0), (0.0, -0.0, 1.0, 0.0)),
    ),
    (
        "Crank",
        "mesh",
        "Crank/Gear_Small_Lower_Spacer_Thick",
        "Gear Small Lower Spacer Thick.obj",
        ((1.0, 0.0, 0.0, 0.0), (-0.0, 1.0, 0.0, 0.0), (0.0, -0.0, 1.0, 0.0)),
    ),
    (
        "Crank",
        "mesh",
        "Crank/Crank",
        "Crank.obj",
        ((1.0, 0.0, 0.0, 0.0), (-0.0, 1.0, 0.0, -0.0), (0.0, -0.0, 1.0, -0.00075)),
    ),
    (
        "Crank",
        "cylinder",
        "Crank/Cylinder_40",
        (-0.091921361, -0.157673445, -0.003420315, 0.000361665039, -2.1832977e-05, -0.308028121056, 0.95137718354),
        (0.0033, 0.029999999),
    ),
    (
        "HummerBody",
        "mesh",
        "HummerBody/Hummer_Back",
        "Hummer Back.obj",
        ((1.0, 0.0, -0.0, -0.0), (-0.0, 1.0, -0.0, 0.0), (-0.0, -0.0, 1.0, 0.0)),
    ),
    (
        "HummerBody",
        "mesh",
        "HummerBody/Hummer_Body_Spacer__2x_",
        "Hummer Body Spacer (2x).obj",
        ((1.0, 0.0, -0.0, -0.0), (-0.0, 1.0, -0.0, 0.0), (-0.0, -0.0, 1.0, 0.0)),
    ),
    (
        "HummerBody",
        "mesh",
        "HummerBody/Hummer_Mount__2x_",
        "Hummer Mount (2x).obj",
        ((1.0, 0.0, -0.0, -0.0), (-0.0, 1.0, -0.0, 0.0), (-0.0, -0.0, 1.0, -0.000375)),
    ),
    (
        "HummerBody",
        "mesh",
        "HummerBody/Hummer_Mount__2x__01",
        "Hummer Mount (2x).obj",
        ((1.0, 0.0, -0.0, -0.0), (-0.0, 1.0, -0.0, 0.0), (-0.0, -0.0, 1.0, 0.00975)),
    ),
    (
        "HummerBody",
        "mesh",
        "HummerBody/Hummer_Body_Spacer__2x__01",
        "Hummer Body Spacer (2x).obj",
        ((1.0, 0.0, -0.0, -0.0), (-0.0, 1.0, -0.0, 0.0), (-0.0, -0.0, 1.0, -0.016875)),
    ),
    (
        "HummerBody",
        "mesh",
        "HummerBody/Wing_Frame",
        "Wing Frame.obj",
        ((1.0, 0.0, -0.0, -0.0), (-0.0, 1.0, -0.0, 0.0), (-0.0, -0.0, 1.0, 0.0)),
    ),
    (
        "HummerBody",
        "mesh",
        "HummerBody/Wing_Frame_Support_Right",
        "Wing Frame Support Right.obj",
        ((1.0, 0.0, -0.0, -0.0), (-0.0, 1.0, -0.0, 0.0), (-0.0, -0.0, 1.0, 0.0)),
    ),
    (
        "HummerBody",
        "mesh",
        "HummerBody/Wing_Frame_Support_Left",
        "Wing Frame Support Left.obj",
        ((1.0, 0.0, -0.0, -0.0), (-0.0, 1.0, -0.0, 0.0), (-0.0, -0.0, 1.0, 0.0)),
    ),
    (
        "HummerBody",
        "mesh",
        "HummerBody/Hummer_Body_Right",
        "Hummer Body Right.obj",
        ((1.0, 0.0, -0.0, -0.0), (-0.0, 1.0, -0.0, 0.0), (-0.0, -0.0, 1.0, 0.0)),
    ),
    (
        "HummerBody",
        "mesh",
        "HummerBody/Hummer_Body_Cam_Link",
        "Hummer Body Cam Link.obj",
        ((1.0, 0.0, -0.0, -0.0), (-0.0, 1.0, -0.0, 0.0), (-0.0, -0.0, 1.0, 0.0)),
    ),
    (
        "HummerBody",
        "mesh",
        "HummerBody/Hummer_Body_Cam_Link_Spacer_A__2x_",
        "Hummer Body Cam Link Spacer A (2x).obj",
        ((1.0, 0.0, -0.0, -0.0), (-0.0, 1.0, -0.0, 0.0), (-0.0, -0.0, 1.0, 0.0)),
    ),
    (
        "HummerBody",
        "mesh",
        "HummerBody/Hummer_Body_Cam_Link_Spacer_A__2x__01",
        "Hummer Body Cam Link Spacer A (2x).obj",
        ((1.0, 0.0, -0.0, -0.0), (-0.0, 1.0, -0.0, 0.0), (-0.0, -0.0, 1.0, -0.0075)),
    ),
    (
        "HummerBody",
        "mesh",
        "HummerBody/Hummer_Body_Left",
        "Hummer Body Left.obj",
        ((1.0, 0.0, -0.0, -0.0), (-0.0, 1.0, -0.0, 0.0), (-0.0, -0.0, 1.0, 0.0)),
    ),
    (
        "HummerBody",
        "cylinder",
        "HummerBody/Cylinder_30",
        (-0.000984539, -0.05516805, 0.0191043, 1.71e-10, 1.51e-10, 3.42e-10, 1.0),
        (0.003, 0.013499999),
    ),
    (
        "HummerBody",
        "mesh",
        "HummerBody/Tail_Fan_Gear",
        "Tail Fan Gear.obj",
        ((1.0, 0.0, -0.0, -0.0), (-0.0, 1.0, -0.0, 0.0), (-0.0, -0.0, 1.0, 0.0)),
    ),
    (
        "HummerBody",
        "cylinder",
        "HummerBody/Cylinder_43",
        (-0.043277486, 0.012477345, -1.993e-06, 0.000582389278, 0.001186890431, 0.39801679878, 0.917377174342),
        (0.00225, 0.01425),
    ),
    (
        "HummerBody",
        "cylinder",
        "HummerBody/Cylinder_44",
        (-0.073854036, -0.03818285, 5.0373e-05, 0.000582389278, 0.001186890431, 0.39801679878, 0.917377174342),
        (0.001125, 0.01425),
    ),
    (
        "HummerBody",
        "cylinder",
        "HummerBody/Cylinder_45",
        (-0.031615775, 0.022592029, -2.3701e-05, 0.000582389278, 0.001186890431, 0.39801679878, 0.917377174342),
        (0.001125, 0.014625),
    ),
    (
        "GearedSpinner",
        "mesh",
        "GearedSpinner/Spinner_Link_Thick",
        "Spinner Link Thick.obj",
        ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, -0.0), (-0.0, 0.0, 1.0, 0.00075)),
    ),
    (
        "GearedSpinner",
        "mesh",
        "GearedSpinner/Spinner_Link_Thin",
        "Spinner Link Thin.obj",
        ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (-0.0, 0.0, 1.0, -0.00075)),
    ),
    (
        "GearedSpinner",
        "mesh",
        "GearedSpinner/Gear_Small_Upper",
        "Gear Small Upper.obj",
        ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (-0.0, 0.0, 1.0, -0.00075)),
    ),
    (
        "GearedSpinner",
        "cylinder",
        "GearedSpinner/Cylinder_31",
        (-5.898e-06, -5.1649e-05, -5.139e-06, -0.000621862035, -0.000768534054, -0.666029313812, 0.745924912968),
        (0.001875, 0.015375),
    ),
    (
        "GearedSpinner",
        "cylinder",
        "GearedSpinner/Cylinder_32",
        (0.017169014, -4.3265e-05, 0.012959318, -0.000621862035, -0.000768534054, -0.666029313812, 0.745924912968),
        (0.001875, 0.006),
    ),
    (
        "GearedSpinner",
        "cylinder",
        "GearedSpinner/Cylinder_33",
        (0.017176853, -4.095e-05, -0.015327211, -0.000621862035, -0.000768534054, -0.666029313812, 0.745924912968),
        (0.001875, 0.00375),
    ),
    (
        "CamFollowerBody",
        "mesh",
        "CamFollowerBody/Cam_Follower_Body",
        "Cam Follower Body.obj",
        ((1.0, -0.0, -0.0, -0.0), (-0.0, 1.0, 0.0, 0.0), (0.0, -0.0, 1.0, 0.00225)),
    ),
    (
        "CamFollowerBody",
        "cylinder",
        "CamFollowerBody/Cylinder_31",
        (0.01783455, -0.0644313, 0.02501115, -1.3e-10, 3.98e-10, 3.64e-10, 1.0),
        (0.0045, 0.00675),
    ),
    (
        "Tail_Feather_A__2x_",
        "mesh",
        "Tail/Tail_Feather_A__2x_",
        "Tail Feather A (2x).obj",
        ((1.0, -0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)),
    ),
    (
        "Tail_Feather_B__2x_",
        "mesh",
        "Tail/Tail_Feather_B__2x_",
        "Tail Feather B (2x).obj",
        ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, -0.0, 0.0), (-0.0, 0.0, 1.0, 0.0)),
    ),
    (
        "Tail_Feather_C",
        "mesh",
        "Tail/Tail_Feather_C",
        "Tail Feather C.obj",
        ((1.0, 0.0, -0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, -0.0)),
    ),
    (
        "Tail_Feather_A__2x__mirrored",
        "mesh",
        "Tail/Tail_Feather_A__2x__mirrored",
        "Tail_Feather_A__2x__mirrored.usd.obj",
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)),
    ),
    (
        "Tail_Feather_B__2x__mirrored",
        "mesh",
        "Tail/Tail_Feather_B__2x__mirrored",
        "Tail_Feather_B__2x__mirrored.usd.obj",
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)),
    ),
    (
        "CamWheelHead",
        "mesh",
        "CamWheelHead/Cam_Wheel_Head",
        "Cam Wheel Head.obj",
        ((1.0, -0.0, -0.0, 0.0), (-0.0, 1.0, 0.0, -0.0), (-0.0, 0.0, 1.0, 0.0015)),
    ),
    (
        "CamWheelHead",
        "mesh",
        "CamWheelHead/Cam_Wheel_Head_Spacer",
        "Cam Wheel Head Spacer.obj",
        ((1.0, -0.0, -0.0, 0.0), (-0.0, 1.0, 0.0, -0.0), (-0.0, 0.0, 1.0, -0.006)),
    ),
    (
        "CamWheelHead",
        "mesh",
        "CamWheelHead/Cam_Wheel_Body",
        "Cam Wheel Body.obj",
        ((1.0, -0.0, -0.0, 0.0), (-0.0, 1.0, 0.0, -0.0), (-0.0, 0.0, 1.0, 0.0015)),
    ),
    (
        "CamWheelHead",
        "mesh",
        "CamWheelHead/Cam_Wheel_Body_Spacer",
        "Cam Wheel Body Spacer.obj",
        ((1.0, -0.0, -0.0, 0.0), (-0.0, 1.0, 0.0, -0.0), (-0.0, 0.0, 1.0, 0.0015)),
    ),
    (
        "CamWheelHead",
        "cylinder",
        "CamWheelHead/Cylinder_30",
        (0.04926465, -0.028557, 0.008720775, -4.35e-10, -2.76e-10, 3.08e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "CamWheelHead",
        "cylinder",
        "CamWheelHead/Cylinder_31",
        (0.0260631, -0.03818265, 0.008720775, -4.35e-10, -2.76e-10, 3.08e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "CamWheelHead",
        "cylinder",
        "CamWheelHead/Cylinder_32",
        (0.03559035, -0.0614007, 0.008720775, -4.35e-10, -2.76e-10, 3.08e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "CamWheelHead",
        "cylinder",
        "CamWheelHead/Cylinder_33",
        (0.0588864, -0.05183025, 0.008720775, -4.35e-10, -2.76e-10, 3.08e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Gear_Large__3x_02",
        "mesh",
        "Gear_Large__3x_02/Gear_Large__3x_02",
        "Gear Large (3x).obj",
        ((1.0, -0.0, -0.0, -0.0), (0.0, 1.0, -0.0, -0.0), (0.0, -0.0, 1.0, -0.0)),
    ),
    (
        "Gear_Large__3x_02",
        "mesh",
        "Gear_Large__3x_02/Hypocycloid_Cam__3x_02",
        "Hypocycloid Cam (3x).obj",
        ((1.0, -0.0, -0.0, -0.0), (0.0, 1.0, -0.0, -0.0), (0.0, -0.0, 1.0, -0.0)),
    ),
    (
        "Gear_Large__3x_01",
        "mesh",
        "Gear_Large__3x_01/Gear_Large__3x_01",
        "Gear Large (3x).obj",
        ((1.0000004768, 0.0, 0.0, -0.0), (0.0, 1.0000004768, 0.0, 0.0), (-0.0, -0.0, 1.0, -0.0)),
    ),
    (
        "Gear_Large__3x_01",
        "mesh",
        "Gear_Large__3x_01/Hypocycloid_Cam__3x_01",
        "Hypocycloid Cam (3x).obj",
        (
            (-0.9862860279, -0.1650479472, 0.0, 0.0765000011),
            (0.1650479472, -0.9862860279, 0.0, -0.096423),
            (0.0, 0.0, 1.0, -0.0),
        ),
    ),
    (
        "Gear_Large__3x_00",
        "mesh",
        "Gear_Large__3x_00/Gear_Large__3x_0",
        "Gear Large (3x).obj",
        ((1.0, 0.0, -0.0, 0.0), (0.0, 1.0, -0.0, 0.0), (0.0, 0.0, 1.0, -0.0)),
    ),
    (
        "Gear_Large__3x_00",
        "mesh",
        "Gear_Large__3x_00/Hypocycloid_Cam__3x_0",
        "Hypocycloid Cam (3x).obj",
        ((1.0, 0.0, -0.0, 0.0), (0.0, 1.0, -0.0, 0.0), (0.0, 0.0, 1.0, -0.0)),
    ),
    (
        "CamWheelBottom",
        "mesh",
        "CamWheelBottom/Cam_Wheel_Bottom",
        "Cam Wheel Bottom.obj",
        ((1.0, 0.0, -0.0, 0.0), (-0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0015)),
    ),
    (
        "CamWheelBottom",
        "mesh",
        "CamWheelBottom/Cam_Wheel_Bottom_Spacer",
        "Cam Wheel Bottom Spacer.obj",
        ((1.0, 0.0, -0.0, 0.0), (-0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0015)),
    ),
    (
        "CamWheelBottom",
        "cylinder",
        "CamWheelBottom/Cylinder_38",
        (-0.023210333, -0.141292147, 0.008720468, 4.12e-10, 2.82e-10, 9.8e-11, 1.0),
        (0.00075, 0.00525),
    ),
    (
        "CamWheelBottom",
        "cylinder",
        "CamWheelBottom/Cylinder_39",
        (-0.046419301, -0.150498361, 0.008720646, 4.12e-10, 2.82e-10, 9.8e-11, 1.0),
        (0.00075, 0.00525),
    ),
    (
        "CamWheelBottom",
        "cylinder",
        "CamWheelBottom/Cylinder_40",
        (-0.037078539, -0.17385315, 0.008720679, 4.12e-10, 2.82e-10, 9.8e-11, 1.0),
        (0.00075, 0.00525),
    ),
    (
        "CamWheelBottom",
        "cylinder",
        "CamWheelBottom/Cylinder_41",
        (-0.013728352, -0.164730454, 0.008720142, 4.12e-10, 2.82e-10, 9.8e-11, 1.0),
        (0.00075, 0.00525),
    ),
    (
        "CamWheelTail",
        "mesh",
        "CamWheelTail/Cam_Wheel_Tail",
        "Cam Wheel Tail.obj",
        ((1.0, -4.5e-07, -0.0, 0.0), (4.5e-07, 1.0, 0.0, 0.0), (-0.0, -0.0, 1.0, 0.0015)),
    ),
    (
        "CamWheelTail",
        "mesh",
        "CamWheelTail/Cam_Wheel_Tail_Spacer",
        "Cam Wheel Tail Spacer.obj",
        ((1.0, -4.5e-07, -0.0, 0.0), (4.5e-07, 1.0, 0.0, 0.0), (-0.0, -0.0, 1.0, 0.0015)),
    ),
    (
        "CamWheelTail",
        "cylinder",
        "CamWheelTail/Cylinder_34",
        (0.05431275, -0.11135955, 0.008720775, 3.14e-10, -1.7e-11, -2e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "CamWheelTail",
        "cylinder",
        "CamWheelTail/Cylinder_35",
        (0.0628749, -0.13487565, 0.008720775, 3.14e-10, -1.7e-11, -2e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "CamWheelTail",
        "cylinder",
        "CamWheelTail/Cylinder_36",
        (0.03931845, -0.1434777, 0.008720775, 3.14e-10, -1.7e-11, -2e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "CamWheelTail",
        "cylinder",
        "CamWheelTail/Cylinder_37",
        (0.03064755, -0.1199022, 0.008720775, 3.14e-10, -1.7e-11, -2e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "TailRack",
        "mesh",
        "TailRack/Tail_Rack",
        "Tail Rack.obj",
        ((1.0, -0.0, 0.0, -0.0), (0.0, 1.0, 0.0, 0.0), (-0.0, 0.0, 1.0, 0.0)),
    ),
    (
        "TailRack",
        "cylinder",
        "TailRack/Cylinder_34",
        (-0.11824575, -0.0518682, 0.000975, 0.702903940567, 0.07698084494, -0.076980844373, -0.702903940413),
        (0.000825, 0.011249972),
    ),
    (
        "CamFollower",
        "cylinder",
        "CamFollower/Cylinder_32",
        (0.0563349, -0.10455885, 0.016171058, -2.27e-10, -1.47e-10, -2e-12, 1.0),
        (0.00375, 0.006),
    ),
    (
        "CamFollower",
        "mesh",
        "CamFollower/CamFollowerTail",
        "CamFollowerTail.usd.obj",
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)),
    ),
    (
        "CamFollowerHead",
        "mesh",
        "CamFollowerHead/Cam_Follower_Head",
        "Cam Follower Head.obj",
        ((1.0, 0.0, -0.0, -0.0), (-0.0, 1.0, 0.0, 0.2176155), (0.0, -0.0, 1.0, -0.0)),
    ),
    (
        "HummerHead",
        "mesh",
        "HummerHead/Hummer_Head_Right",
        "Hummer Head Right.obj",
        ((1.0, 0.0, -0.0, -0.0), (0.0, 1.0, 0.0, 0.0), (-0.0, 0.0, 1.0, 0.0)),
    ),
    (
        "HummerHead",
        "mesh",
        "HummerHead/Hummer_Head_Left",
        "Hummer Head Left.obj",
        ((1.0, 0.0, -0.0, -0.0), (0.0, 1.0, 0.0, 0.0), (-0.0, 0.0, 1.0, 0.0)),
    ),
    (
        "HummerHead",
        "mesh",
        "HummerHead/Hummer_Bill",
        "Hummer Bill.obj",
        ((1.0, 0.0, -0.0, -0.0), (0.0, 1.0, 0.0, 0.0), (-0.0, 0.0, 1.0, 0.0)),
    ),
    (
        "HummerHead",
        "cylinder",
        "HummerHead/Cylinder_33",
        (0.06418095, 0.011122065, 0.004673775, -2.44e-10, -7.2e-11, -3.21e-10, 1.0),
        (0.003, 0.00825),
    ),
    (
        "WingLinkArcLeft",
        "mesh",
        "WingLinkArcLeft/Wing_Link_Arc__2x__1",
        "Wing Link Arc (2x).obj",
        ((1.0, 0.0, -0.0, 0.0), (-0.0, 1.0, 0.0, -0.0), (-0.0, -0.0, 1.0, -0.03375)),
    ),
    (
        "WingLinkArcLeft",
        "cylinder",
        "WingLinkArcLeft/Cylinder_32",
        (0.015840648, 0.018229437, -0.019599986, 0.004199882815, -0.003532887947, 0.02265886994, 0.99972819071),
        (0.00225, 0.00375),
    ),
    (
        "WingLinkArcLeft",
        "cylinder",
        "WingLinkArcLeft/Cylinder_33",
        (-0.007751555, 0.029371983, -0.01575, 0.004199882815, -0.003532887947, 0.02265886994, 0.99972819071),
        (0.00225, 0.0045),
    ),
    (
        "WingLinkArcRight",
        "mesh",
        "WingLinkArcRight/Wing_Link_Arc__2x_",
        "Wing Link Arc (2x).obj",
        ((1.0, 0.0, -0.0, 0.0), (0.0, 1.0, 0.0, 0.2176155), (-0.0, 0.0, 1.0, 0.00075)),
    ),
    (
        "WingLinkArcRight",
        "cylinder",
        "WingLinkArcRight/Cylinder_31",
        (-0.00766492, 0.24692859, 0.014707787, 0.001677565954, 0.00054480309, 0.019706222345, 0.999804257724),
        (0.00225, 0.0045),
    ),
    (
        "WingLinkArcRight",
        "cylinder",
        "WingLinkArcRight/Cylinder_32",
        (0.016773347, 0.235460863, 0.019180656, 0.001677565954, 0.00054480309, 0.019706222345, 0.999804257724),
        (0.00225, 0.00375),
    ),
    (
        "WingLinkStraightRight",
        "mesh",
        "WingLinkStraightRight/Wing_Link_Straight__2x_",
        "Wing Link Straight (2x).obj",
        ((1.0, -0.0, -0.0, -0.0), (0.0, 1.0, -0.0, -0.0), (0.0, 0.0, 1.0, 0.000375)),
    ),
    (
        "WingLinkStraightLeft",
        "mesh",
        "WingLinkStraightLeft/Wing_Link_Straight__2x__1",
        "Wing Link Straight (2x).obj",
        ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, -0.0, -0.0), (0.0, 0.0, 1.0, -0.0255)),
    ),
    (
        "WingRight",
        "mesh",
        "WingRight/Wing__2x_",
        "Wing (2x).obj",
        ((1.0, 0.0, 0.0, 0.0), (-0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, -0.0)),
    ),
    (
        "WingRight",
        "mesh",
        "WingRight/Shoulder_Thick__2x_",
        "Shoulder Thick (2x).obj",
        ((1.0, 0.0, 0.0, 0.0), (-0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, -0.0)),
    ),
    (
        "WingRight",
        "mesh",
        "WingRight/Shoulder_Thin__2x_",
        "Shoulder Thin (2x).obj",
        ((1.0, 0.0, 0.0, 0.0), (-0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, -0.0)),
    ),
    (
        "WingRight",
        "mesh",
        "WingRight/Pivot_Block__2x_",
        "Pivot Block (2x).obj",
        ((1.0, 0.0, 0.0, 0.0), (-0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, -0.0)),
    ),
    (
        "ShoulderRight",
        "mesh",
        "ShoulderRight/Shoulder_Pivot__2x_",
        "Shoulder Pivot (2x).obj",
        ((1.0, 0.0, 0.0, -0.0), (-0.0, 1.0, 0.0, 0.0), (-0.0, 0.0, 1.0, 0.0)),
    ),
    (
        "WingRightConnector",
        "cylinder",
        "WingRightConnector/Cylinder_34",
        (0.00442011, 0.01162554, 0.0238443, 0.113423291838, 0.838061859101, -0.178565706654, 0.502893393868),
        (0.001875003, 0.013500016),
    ),
    (
        "WingRightConnector",
        "mesh",
        "WingRightConnector/Sphere",
        "Sphere.usd.obj",
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)),
    ),
    (
        "WingLeft",
        "mesh",
        "WingLeft/Shoulder_Thick__2x__mirrored",
        "Shoulder_Thick__2x__mirrored.usd.obj",
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)),
    ),
    (
        "WingLeft",
        "mesh",
        "WingLeft/Shoulder_Thin__2x__mirrored",
        "Shoulder_Thin__2x__mirrored.usd.obj",
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)),
    ),
    (
        "WingLeft",
        "mesh",
        "WingLeft/Wing__2x__mirrored",
        "Wing__2x__mirrored.usd.obj",
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)),
    ),
    (
        "WingLeft",
        "mesh",
        "WingLeft/Pivot_Block__2x__mirrored",
        "Pivot_Block__2x__mirrored.usd.obj",
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)),
    ),
    (
        "ShoulderLeft",
        "mesh",
        "ShoulderLeft/Shoulder_Pivot__2x__mirrored",
        "Shoulder_Pivot__2x__mirrored.usd.obj",
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)),
    ),
    (
        "WingLeftConnector",
        "cylinder",
        "WingLeftConnector/Cylinder_35",
        (0.00442011, 0.2292405, -0.0238443, 0.184355088654, 0.500834870482, -0.103748020383, 0.839293740036),
        (0.001875002, 0.013500034),
    ),
    (
        "WingLeftConnector",
        "mesh",
        "WingLeftConnector/Sphere_01",
        "Sphere_01.usd.obj",
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)),
    ),
    (
        "TailMount",
        "mesh",
        "TailMount/Tail_Mount_Left",
        "Tail Mount Left.obj",
        ((1.0, -0.0, -0.0, -0.0), (-0.0, 1.0, 0.0, 0.0), (0.0, -0.0, 1.0, 0.0)),
    ),
    (
        "TailMount",
        "mesh",
        "TailMount/Tail_Mount_Right",
        "Tail Mount Right.obj",
        ((1.0, -0.0, -0.0, -0.0), (-0.0, 1.0, 0.0, 0.0), (0.0, -0.0, 1.0, 0.0)),
    ),
    (
        "TailMount",
        "mesh",
        "TailMount/Tail_Feathers_Mount",
        "Tail Feathers Mount.obj",
        ((0.9997270184, 0.0256978813, -0.0, -0.0), (-0.0233617083, 1.0996998097, 0.0, 0.003), (0.0, 0.0, 1.0, 0.0)),
    ),
    (
        "TailMount",
        "cylinder",
        "TailMount/Cylinder_32",
        (-0.08574735, -0.06118035, -0.001008251, -1.42e-10, 2.9e-10, 3.12e-10, 1.0),
        (0.003, 0.009),
    ),
    (
        "TailMount",
        "cylinder",
        "TailMount/Cylinder_33",
        (-0.0413226, -0.0516057, -0.00100825, -1.42e-10, 2.9e-10, 3.12e-10, 1.0),
        (0.003, 0.009),
    ),
    (
        "TailMount",
        "cylinder",
        "TailMount/Cylinder_34",
        (-0.05213115, -0.07236015, 0.014819625, -1.42e-10, 2.9e-10, 3.12e-10, 1.0),
        (0.003375, 0.00975),
    ),
    (
        "TailMount",
        "mesh",
        "TailMount/Tail_Mount_Left_Standoff_A",
        "Tail Mount Left Standoff A.obj",
        ((1.0, -0.0, -0.0, -0.0), (-0.0, 1.0, 0.0, 0.0), (0.0, -0.0, 1.0, 0.0)),
    ),
    (
        "TailMount",
        "mesh",
        "TailMount/Tail_Mount_Left_Standoff_B",
        "Tail Mount Left Standoff B.obj",
        ((1.0, -0.0, -0.0, -0.0), (-0.0, 1.0, 0.0, 0.0), (0.0, -0.0, 1.0, 0.0)),
    ),
    (
        "TailPinion",
        "mesh",
        "TailPinion/Tail_Pinion_Thick",
        "Tail Pinion Thick.obj",
        ((1.0, 0.0, 0.0, 0.0), (-0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)),
    ),
    (
        "TailPinion",
        "mesh",
        "TailPinion/Tail_Pinion_Thin",
        "Tail Pinion Thin.obj",
        ((1.0, 0.0, 0.0, 0.0), (-0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)),
    ),
    (
        "Frame",
        "mesh",
        "Frame/FrameMesh",
        "Frame.obj",
        ((1.0, -0.0, -0.0, 0.0), (0.0, 1.0, -0.0, 0.0), (0.0, 0.0, 1.0, 0.0)),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder",
        (0.04238445, -0.07520145, 0.0026778, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_01",
        (0.0247926, -0.0692712, 0.0026778, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_02",
        (0.013834905, -0.0541392, 0.0026778, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_03",
        (0.014010045, -0.03562425, 0.0026778, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_04",
        (0.02481645, -0.0206739, 0.0026778, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_05",
        (0.0424302, -0.01502595, 0.0026778, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_06",
        (0.0600132, -0.02065245, 0.0026778, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_07",
        (0.0710352, -0.03566355, 0.0026778, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_08",
        (0.07094505, -0.0542181, 0.0026778, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_09",
        (0.06015315, -0.0692466, 0.0026778, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_10",
        (-0.030131348, -0.127176322, 0.002676175, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_11",
        (-0.047690763, -0.132923367, 0.002677001, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_12",
        (-0.058739878, -0.148137054, 0.002675621, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_13",
        (-0.058969264, -0.166893516, 0.00267539, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_14",
        (-0.047646529, -0.181927235, 0.002677975, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_15",
        (-0.029827823, -0.187665387, 0.002679387, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_16",
        (-0.01216435, -0.181891461, 0.002678688, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_17",
        (-0.001284871, -0.166917513, 0.002679407, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_18",
        (-0.001290229, -0.148123078, 0.002678336, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_19",
        (-0.012378775, -0.133023909, 0.002677676, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_20",
        (0.06609765, -0.1044387, 0.0026778, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_21",
        (0.04894035, -0.0974598, 0.0026778, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_22",
        (0.03091095, -0.10178775, 0.0026778, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_23",
        (0.01911645, -0.11607705, 0.0026778, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_24",
        (0.01763385, -0.13454625, 0.0026778, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_25",
        (0.02752785, -0.15045, 0.0026778, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_26",
        (0.0447123, -0.157401, 0.0026778, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_27",
        (0.0627441, -0.152892, 0.0026778, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_28",
        (0.0746403, -0.13878825, 0.0026778, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinders/Cylinder_29",
        (0.0759624, -0.12016155, 0.0026778, -1.12e-10, 1.31e-10, 3.79e-10, 1.0),
        (0.000675, 0.00525),
    ),
    (
        "Frame",
        "mesh",
        "Frame/Counterweight_Flange",
        "Counterweight Flange.obj",
        ((1.0, -0.0, -0.0, 0.0), (0.0, 1.0, -0.0, 0.0), (0.0, 0.0, 1.0, 0.0)),
    ),
    (
        "Frame",
        "mesh",
        "Frame/Counterweight_Arm",
        "Counterweight Arm.obj",
        ((1.0, -0.0, -0.0, 0.0), (0.0, 1.0, -0.0, 0.0), (0.0, 0.0, 1.0, 0.0)),
    ),
    (
        "Frame",
        "mesh",
        "Frame/Support_Frame",
        "Support Frame.obj",
        ((1.0, -0.0, -0.0, 0.0), (0.0, 1.0, -0.0, 0.0), (0.0, 0.0, 1.0, 0.0)),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinder",
        (-0.150918, -0.16584915, 0.006607815, 0.702903970068, -0.07698057547, 0.076980575302, -0.702903969891),
        (0.019124963, 0.018750002),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinder_35",
        (0.093550826, -0.102076411, 0.003889624, 0.000149141026, 0.000282944511, 0.425677816755, 0.90487473941),
        (0.0024, 0.02025),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinder_36",
        (0.017688956, -0.087918908, 0.007591699, 0.000149141026, 0.000282944511, 0.425677816755, 0.90487473941),
        (0.0024, 0.024),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinder_37",
        (0.028244657, 0.001739352, -0.001907484, 0.000149141026, 0.000282944511, 0.425677816755, 0.90487473941),
        (0.00225, 0.01425),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinder_38",
        (-0.074143324, -0.134110872, -0.006693578, 0.000149141026, 0.000282944511, 0.425677816755, 0.90487473941),
        (0.00225, 0.00975),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinder_39",
        (-8.7653e-05, -0.19630189, -0.006703789, 0.000149141026, 0.000282944511, 0.425677816755, 0.90487473941),
        (0.00225, 0.00975),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinder_40",
        (0.042437855, -0.045050342, 0.004645867, 0.000149141026, 0.000282944511, 0.425677816755, 0.90487473941),
        (0.001275, 0.021),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinder_41",
        (0.04691127, -0.127628916, 0.000797225, 0.000149141026, 0.000282944511, 0.425677816755, 0.90487473941),
        (0.001275, 0.01725),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinder_42",
        (-0.0303729, -0.157969663, 0.000795397, 0.000149141026, 0.000282944511, 0.425677816755, 0.90487473941),
        (0.001275, 0.01725),
    ),
    (
        "Frame",
        "cylinder",
        "Frame/Cylinder_01",
        (-0.149095303, -0.155836892, 0.006609077, 0.702903970068, -0.07698057547, 0.076980575302, -0.702903969891),
        (0.0021, 0.018750002),
    ),
    (
        "FrameGround",
        "mesh",
        "FrameGround/Frame_Support_A",
        "Frame Support A.obj",
        ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (-0.0, 0.0, 1.0, 0.0)),
    ),
    (
        "FrameGround",
        "mesh",
        "FrameGround/Frame_Support_B",
        "Frame Support B.obj",
        ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (-0.0, 0.0, 1.0, 0.0)),
    ),
    (
        "FrameGround",
        "mesh",
        "FrameGround/Base",
        "Base.obj",
        ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (-0.0, 0.0, 1.0, 0.0)),
    ),
    (
        "FrameGround",
        "cylinder",
        "FrameGround/Cylinder_33",
        (-0.036, -0.18375, 0.018, -3.19e-10, -5.88e-10, 8.8e-11, 1.0),
        (0.00375, 0.009),
    ),
    (
        "FrameGround",
        "cylinder",
        "FrameGround/Cylinder_34",
        (-0.021, -0.18375, 0.018, -3.19e-10, -5.88e-10, 8.8e-11, 1.0),
        (0.00375, 0.009),
    ),
    (
        "FrameGround",
        "cylinder",
        "FrameGround/Cylinder_35",
        (-0.006, -0.183, 0.018, -3.19e-10, -5.88e-10, 8.8e-11, 1.0),
        (0.00525, 0.009),
    ),
    (
        "FrameGround",
        "cylinder",
        "FrameGround/Cylinder_40",
        (-0.063024805, -0.199235352, 7.731e-05, 4.6130303e-05, 0.000306411626, 0.535103467786, 0.844786471688),
        (0.00225, 0.027),
    ),
    (
        "FrameGround",
        "cylinder",
        "FrameGround/Cylinder_41",
        (0.091865928, -0.199235352, 7.731e-05, 4.6130303e-05, 0.000306411626, 0.535103467786, 0.844786471688),
        (0.00225, 0.027),
    ),
    (
        "FrameGround",
        "cylinder",
        "FrameGround/Cylinder_42",
        (0.118715324, -0.126976304, 7.731e-05, 4.6130303e-05, 0.000306411626, 0.535103467786, 0.844786471688),
        (0.00225, 0.027),
    ),
    (
        "FrameGround",
        "cylinder",
        "FrameGround/Cylinder_43",
        (0.125872477, -0.113923969, 7.731e-05, 4.6130303e-05, 0.000306411626, 0.535103467786, 0.844786471688),
        (0.00225, 0.027),
    ),
    (
        "FrameGround",
        "cylinder",
        "FrameGround/Cylinder_44",
        (-0.03506559, -0.182202379, 0.019581214, 4.6130303e-05, 0.000306411626, 0.535103467786, 0.844786471688),
        (0.00225, 0.0075),
    ),
    (
        "FrameGround",
        "mesh",
        "FrameGround/Flower/Flower_Center_01",
        "Flower Center.obj",
        ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (-0.0, 0.0, 1.0, 0.0)),
    ),
    (
        "FrameGround",
        "mesh",
        "FrameGround/Flower/Flower_Petal/Flower_Petal_Front__5x__01",
        "Flower Petal Front (5x).obj",
        ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (-0.0, 0.0, 1.0, 0.0)),
    ),
    (
        "FrameGround",
        "mesh",
        "FrameGround/Flower/Flower_Petal/Flower_Petal_Rear__5x__01",
        "Flower Petal Rear (5x).obj",
        ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (-0.0, 0.0, 1.0, 0.0)),
    ),
    (
        "FrameGround",
        "mesh",
        "FrameGround/Flower/Flower_Petal_01/Flower_Petal_Front__5x__01",
        "Flower Petal Front (5x).obj",
        ((1.0, 0.0, -0.0, 0.0), (0.0, 0.3090170151, -0.9510570109, 0.0414), (-0.0, 0.9510570109, 0.3090170151, -0.057)),
    ),
    (
        "FrameGround",
        "mesh",
        "FrameGround/Flower/Flower_Petal_01/Flower_Petal_Rear__5x__01",
        "Flower Petal Rear (5x).obj",
        ((1.0, 0.0, -0.0, 0.0), (0.0, 0.3090170151, -0.9510570109, 0.0414), (-0.0, 0.9510570109, 0.3090170151, -0.057)),
    ),
    (
        "FrameGround",
        "mesh",
        "FrameGround/Flower/Flower_Petal_02/Flower_Petal_Front__5x__01",
        "Flower Petal Front (5x).obj",
        ((1.0, -0.0, 0.0, 0.0), (0.0, 0.3090169944, 0.9510565163, 0.0414), (-0.0, -0.9510565163, 0.3090169944, 0.057)),
    ),
    (
        "FrameGround",
        "mesh",
        "FrameGround/Flower/Flower_Petal_02/Flower_Petal_Rear__5x__01",
        "Flower Petal Rear (5x).obj",
        ((1.0, -0.0, 0.0, 0.0), (0.0, 0.3090169944, 0.9510565163, 0.0414), (-0.0, -0.9510565163, 0.3090169944, 0.057)),
    ),
    (
        "FrameGround",
        "mesh",
        "FrameGround/Flower/Flower_Petal_03/Flower_Petal_Front__5x__01",
        "Flower Petal Front (5x).obj",
        (
            (1.0, 0.0, -0.0, 0.0),
            (0.0, -0.8090170545, -0.5877849668, 0.1086),
            (-0.0, 0.5877849668, -0.8090170545, -0.03525),
        ),
    ),
    (
        "FrameGround",
        "mesh",
        "FrameGround/Flower/Flower_Petal_03/Flower_Petal_Rear__5x__01",
        "Flower Petal Rear (5x).obj",
        (
            (1.0, 0.0, -0.0, 0.0),
            (0.0, -0.8090170545, -0.5877849668, 0.1086),
            (-0.0, 0.5877849668, -0.8090170545, -0.03525),
        ),
    ),
    (
        "FrameGround",
        "mesh",
        "FrameGround/Flower/Flower_Petal_04/Flower_Petal_Front__5x__01",
        "Flower Petal Front (5x).obj",
        (
            (1.0, 0.0, -0.0, 0.0),
            (0.0, -0.8090169944, 0.5877852523, 0.1086),
            (-0.0, -0.5877852523, -0.8090169944, 0.03525),
        ),
    ),
    (
        "FrameGround",
        "mesh",
        "FrameGround/Flower/Flower_Petal_04/Flower_Petal_Rear__5x__01",
        "Flower Petal Rear (5x).obj",
        (
            (1.0, 0.0, -0.0, 0.0),
            (0.0, -0.8090169944, 0.5877852523, 0.1086),
            (-0.0, -0.5877852523, -0.8090169944, 0.03525),
        ),
    ),
    (
        "FrameGround",
        "mesh",
        "FrameGround/Flower/Flower_Stem",
        "Flower Stem.obj",
        ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (-0.0, 0.0, 1.0, 0.0)),
    ),
    (
        "FrameGround",
        "mesh",
        "FrameGround/Flower/Slider/Cube",
        "Cube.usd.obj",
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)),
    ),
    (
        "FrameGround",
        "cylinder",
        "FrameGround/Flower/Slider/Cylinder_44",
        (-0.009457005, -0.189121764, 0.0135, 4.6130303e-05, 0.000306411626, 0.535103467786, 0.844786471688),
        (0.00375, 0.003),
    ),
    (
        "FrameGround",
        "cylinder",
        "FrameGround/Flower/Slider/Cylinder_45",
        (-0.054184552, -0.178528263, 0.0135, 4.6130303e-05, 0.000306411626, 0.535103467786, 0.844786471688),
        (0.00375, 0.003),
    ),
]

JOINTS = [
    (
        "Gear_Large__3x_00",
        "Hypocycloid_Gear__3x_0",
        "revolute",
        "Z",
        (0.042449999, -0.050999994, 0.00075, 0.0, 0.0, 0.0, 1.0),
        (0.042449999, -0.050999994, 0.0, 0.0, 0.0, 0.0, 1.0),
        {},
    ),
    (
        "Gear_Large__3x_02",
        "Hypocycloid_Gear__3x_02",
        "revolute",
        "Z",
        (0.042449999, -0.050850002, 0.00075, 0.0, 0.0, 0.0, 1.0),
        (0.042449999, -0.051000001, 0.0, 0.0, 0.0, 0.0, 1.0),
        {},
    ),
    (
        "Gear_Large__3x_01",
        "Hypocycloid_Gear__3x_01",
        "revolute",
        "Z",
        (0.042999251, -0.039242971, 0.00075, 0.0, 0.0, -0.134850933, 0.990865897),
        (0.042449999, -0.051000001, 0.0, 0.0, 0.0, 0.0, 1.0),
        {},
    ),
    (
        "Frame",
        "Crank",
        "revolute",
        "Z",
        (-0.092085385, -0.157514191, 0.0, -0.0, -0.0, 0.069756472, 0.99756405),
        (-0.091950002, -0.1575, 0.0, 0.0, 0.0, 0.0, 1.0),
        {"target_kd": 5.729577951308232, "target_vel": -3.490658503988659},
    ),
    (
        "Frame",
        "HummerBody",
        "revolute",
        "Z",
        (0.0, -1e-09, 0.0, 0.0, 0.0, 0.0, 1.0),
        (-0.0, -1e-09, 0.0, 0.0, 0.0, 0.0, 1.0),
        {},
    ),
    (
        "Frame",
        "GearedSpinner",
        "revolute",
        "Z",
        (0.0, -1e-09, 0.0, 0.0, 0.0, 0.0, 1.0),
        (0.0, -1e-09, -0.0, 0.0, 0.0, 0.0, 1.0),
        {},
    ),
    (
        "Frame",
        "CamFollowerBody",
        "revolute",
        "Z",
        (0.017549999, -0.088200002, 0.0, 0.0, 0.0, 0.0, 1.0),
        (0.017549999, -0.088200002, 0.0, 0.0, 0.0, 0.0, 1.0),
        {},
    ),
    (
        "TailMount",
        "Tail_Feather_A__2x_",
        "revolute",
        "Y",
        (-0.093567574, -0.028875175, 0.00102731, -0.004446761, -0.009778493, 0.105472701, 0.994364177),
        (-0.093567574, -0.030375177, 0.00102731, -0.004446761, -0.009778493, 0.105472701, 0.994364177),
        {},
    ),
    (
        "TailMount",
        "Tail_Feather_B__2x_",
        "revolute",
        "Y",
        (-0.093567574, -0.028875175, 0.00102731, -0.004446761, -0.009778493, 0.105472701, 0.994364177),
        (-0.093567574, -0.030225177, 0.00102731, -0.004446751, -0.009778472, 0.105472468, 0.994364202),
        {},
    ),
    (
        "TailMount",
        "Tail_Feather_C",
        "revolute",
        "Y",
        (-0.093567574, -0.028875175, 0.00102731, -0.004446761, -0.009778493, 0.105472701, 0.994364177),
        (-0.093567574, -0.030075173, 0.00102731, -0.004446751, -0.009778472, 0.105472468, 0.994364202),
        {},
    ),
    (
        "TailMount",
        "Tail_Feather_A__2x__mirrored",
        "revolute",
        "Y",
        (-0.093567574, -0.028875175, 0.00102731, -0.004446761, -0.009778493, 0.105472701, 0.994364177),
        (-0.095247574, -0.022575175, -0.00084769, -0.004446751, -0.009778472, 0.105472468, 0.994364202),
        {},
    ),
    (
        "TailMount",
        "Tail_Feather_B__2x__mirrored",
        "revolute",
        "Y",
        (-0.093567574, -0.028875175, 0.00102731, -0.004446761, -0.009778493, 0.105472701, 0.994364177),
        (-0.094407578, -0.026325175, -0.00084769, -0.004446751, -0.009778472, 0.105472468, 0.994364202),
        {},
    ),
    (
        "Frame",
        "CamWheelHead",
        "revolute",
        "Z",
        (0.042449999, -0.045, 0.0, 0.0, 0.0, 0.0, 1.0),
        (0.042449999, -0.045, 0.0, 0.0, 0.0, 0.0, 1.0),
        {},
    ),
    (
        "Frame",
        "Gear_Large__3x_02",
        "revolute",
        "Z",
        (-0.03015, -0.157650003, 0.0, 0.0, 0.0, 0.0, 1.0),
        (0.042449999, -0.045, -0.0, 0.0, 0.0, 0.0, 1.0),
        {},
    ),
    (
        "Frame",
        "Gear_Large__3x_01",
        "revolute",
        "Z",
        (0.046894991, -0.127376604, 0.0, 0.0, -0.0, 0.082808205, 0.996565503),
        (0.042450019, -0.045000021, -0.0, -0.0, 0.0, 2.1e-08, 1.0),
        {},
    ),
    (
        "Frame",
        "Gear_Large__3x_00",
        "revolute",
        "Z",
        (0.042449999, -0.045, 0.0, 0.0, 0.0, 0.0, 1.0),
        (0.042449999, -0.045, -0.0, 0.0, 0.0, 0.0, 1.0),
        {},
    ),
    (
        "Frame",
        "CamWheelBottom",
        "revolute",
        "Z",
        (-0.03015, -0.157650003, 0.0, 0.0, 0.0, 0.0, 1.0),
        (-0.03015, -0.157650003, 0.0, 0.0, 0.0, 0.0, 1.0),
        {},
    ),
    (
        "Frame",
        "CamWheelTail",
        "revolute",
        "Z",
        (0.046894991, -0.127376604, 0.0, -0.0, -0.0, 0.082807988, 0.996565521),
        (0.046894991, -0.127376604, 0.0, 0.0, -0.0, 0.082807988, 0.996565521),
        {},
    ),
    (
        "Frame",
        "CamFollower",
        "revolute",
        "Z",
        (0.093374999, -0.102300003, 0.0, 0.0, 0.0, 0.0, 1.0),
        (0.093374999, -0.102300003, -0.0, 0.0, 0.0, 0.0, 1.0),
        {},
    ),
    (
        "Frame",
        "CamFollowerHead",
        "revolute",
        "Z",
        (0.0282, 0.001384505, 0.0, 0.0, 0.0, 0.0, 1.0),
        (0.0282, 0.219000006, -0.0, 0.0, 0.0, 0.0, 1.0),
        {},
    ),
    (
        "HummerBody",
        "HummerHead",
        "revolute",
        "Z",
        (-0.0075, 0.0375, -0.0, 0.0, 0.0, 0.0, 1.0),
        (-0.0075, 0.0375, 0.0, 0.0, 0.0, 0.0, 1.0),
        {},
    ),
    (
        "GearedSpinner",
        "WingLinkArcLeft",
        "revolute",
        "Z",
        (0.017364224, -5.9881e-05, -0.0, 0.0, 0.0, 0.0, 1.0),
        (0.017364224, -5.9881e-05, 0.0, 0.0, 0.0, 0.0, 1.0),
        {},
    ),
    (
        "WingLeftConnector",
        "WingLinkArcLeft",
        "ball",
        "X",
        (0.015, 0.223499994, -0.018000001, 0.0, 0.0, 0.0, 1.0),
        (0.016407979, 0.018289161, -0.023108125, 0.0, 0.0, 0.0, 1.0),
        {},
    ),
    (
        "GearedSpinner",
        "WingLinkArcRight",
        "revolute",
        "Z",
        (0.01732667, -6.1618e-05, -0.0, 0.0, 0.0, 0.0, 1.0),
        (0.01732667, 0.217553887, 0.0, 0.0, 0.0, 0.0, 1.0),
        {},
    ),
    (
        "WingRightConnector",
        "WingLinkArcRight",
        "ball",
        "X",
        (0.015, 0.0072, 0.018000001, 0.0, 0.0, 0.0, 1.0),
        (0.016350001, 0.235949993, 0.0225, 0.0, 0.0, 0.0, 1.0),
        {},
    ),
    (
        "HummerBody",
        "WingLinkStraightRight",
        "revolute",
        "Z",
        (-0.031850077, 0.022709895, 0.0, 0.0, 0.0, 0.0, 1.0),
        (-0.031850077, 0.022709895, 0.0, 0.0, 0.0, 0.0, 1.0),
        {},
    ),
    (
        "WingLinkArcRight",
        "WingLinkStraightRight",
        "revolute",
        "Z",
        (-0.007587115, 0.246683149, 0.0, 0.0, 0.0, 0.0, 1.0),
        (-0.007587115, 0.029067643, 0.0, 0.0, 0.0, 0.0, 1.0),
        {},
    ),
    (
        "HummerBody",
        "WingLinkStraightLeft",
        "revolute",
        "Z",
        (-0.031915745, 0.022515786, 0.0, 0.0, 0.0, 0.0, 1.0),
        (-0.031915745, 0.022515786, -0.0, 0.0, 0.0, 0.0, 1.0),
        {},
    ),
    (
        "WingLinkArcLeft",
        "WingLinkStraightLeft",
        "revolute",
        "Z",
        (-0.007720708, 0.029090794, 0.0, 0.0, 0.0, 0.0, 1.0),
        (-0.007720708, 0.029090794, -0.0, 0.0, 0.0, 0.0, 1.0),
        {},
    ),
    (
        "ShoulderRight",
        "WingRight",
        "revolute",
        "X",
        (-0.010377991, 0.019504643, 0.032190614, -0.05235112, 0.298100764, -0.167824116, 0.938205927),
        (-0.010377991, 0.019504643, 0.032190614, -0.053712581, 0.251267762, -0.162318409, 0.952697331),
        {},
    ),
    (
        "HummerBody",
        "ShoulderRight",
        "revolute",
        "Y",
        (-0.019598697, -1e-09, 0.037406924, 0.0, 0.0, 0.0, 1.0),
        (-0.019598697, -1e-09, 0.037406924, 0.0, 0.0, 0.0, 1.0),
        {},
    ),
    (
        "WingRight",
        "WingRightConnector",
        "revolute",
        "Y",
        (-0.004509423, 0.018489289, 0.029085658, 0.976302795, -0.19440539, -0.03911791, 0.086655562),
        (-0.004509423, 0.018489289, 0.029085658, 0.976302795, -0.19440539, -0.03911791, 0.086655562),
        {},
    ),
    (
        "ShoulderLeft",
        "WingLeft",
        "revolute",
        "X",
        (-0.010377991, 0.019504638, -0.032190599, -0.091470988, -0.21926013, -0.232372497, 0.943165456),
        (-0.010377991, 0.019504638, -0.032190599, -0.091470988, -0.21926013, -0.232372497, 0.943165456),
        {},
    ),
    (
        "HummerBody",
        "ShoulderLeft",
        "revolute",
        "Y",
        (-0.01950597, -1e-09, -0.037517803, 0.0, 0.0, 0.0, 1.0),
        (-0.01950597, -1e-09, -0.037517803, 0.0, 0.0, 0.0, 1.0),
        {},
    ),
    (
        "WingLeft",
        "WingLeftConnector",
        "revolute",
        "Y",
        (-0.004509423, 0.018489289, -0.029085599, 0.97697549, -0.185009358, 0.000371066, -0.106255788),
        (-0.004509423, 0.236104789, -0.029085599, 0.97697549, -0.185009358, 0.000371066, -0.106255788),
        {},
    ),
    (
        "HummerBody",
        "TailMount",
        "revolute",
        "Z",
        (-0.073949997, -0.038249999, 0.0, 0.0, 0.0, 0.0, 1.0),
        (-0.073949997, -0.038249999, 0.0, 0.0, 0.0, 0.0, 1.0),
        {},
    ),
    (
        "TailMount",
        "TailPinion",
        "revolute",
        "Z",
        (-0.043110166, -0.031501036, 0.0, 0.0, 0.0, 0.0, 1.0),
        (-0.043110166, -0.031501036, -0.0, 0.0, 0.0, 0.0, 1.0),
        {},
    ),
    (
        "FrameGround",
        "Frame",
        "revolute",
        "Z",
        (-0.091907609, -0.157658043, -0.0, 0.0, 0.0, 0.0, 1.0),
        (-0.091907609, -0.157658043, 0.0, 0.0, 0.0, 0.0, 1.0),
        {"target_ke": 0.5729577951308232, "target_kd": 0.5729577951308232, "target_pos": 0.3490658503988659},
    ),
]

COLLISION_LABELS = [
    "Hypocycloid_Gear__3x_0/Hypocycloid_Gear__3x_0_mesh",
    "Hypocycloid_Gear__3x_02/Hypocycloid_Gear__3x_0_mesh",
    "Hypocycloid_Gear__3x_01/Hypocycloid_Gear__3x_0_mesh",
    "Crank/Gear_Small_Lower",
    "Crank/Crank",
    "HummerBody/Hummer_Body_Right",
    "HummerBody/Hummer_Body_Left",
    "HummerBody/Cylinder_30",
    "HummerBody/Tail_Fan_Gear",
    "GearedSpinner/Spinner_Link_Thick",
    "GearedSpinner/Spinner_Link_Thin",
    "GearedSpinner/Gear_Small_Upper",
    "CamFollowerBody/Cam_Follower_Body",
    "CamFollowerBody/Cylinder_31",
    "Tail/Tail_Feather_A__2x_",
    "Tail/Tail_Feather_B__2x_",
    "Tail/Tail_Feather_C",
    "Tail/Tail_Feather_A__2x__mirrored",
    "Tail/Tail_Feather_B__2x__mirrored",
    "CamWheelHead/Cam_Wheel_Head",
    "CamWheelHead/Cam_Wheel_Body",
    "CamWheelHead/Cylinder_30",
    "CamWheelHead/Cylinder_31",
    "CamWheelHead/Cylinder_32",
    "CamWheelHead/Cylinder_33",
    "Gear_Large__3x_02/Gear_Large__3x_02",
    "Gear_Large__3x_01/Gear_Large__3x_01",
    "Gear_Large__3x_00/Gear_Large__3x_0",
    "CamWheelBottom/Cam_Wheel_Bottom",
    "CamWheelBottom/Cylinder_38",
    "CamWheelBottom/Cylinder_39",
    "CamWheelBottom/Cylinder_40",
    "CamWheelBottom/Cylinder_41",
    "CamWheelTail/Cam_Wheel_Tail",
    "CamWheelTail/Cylinder_34",
    "CamWheelTail/Cylinder_35",
    "CamWheelTail/Cylinder_36",
    "CamWheelTail/Cylinder_37",
    "TailRack/Tail_Rack",
    "TailRack/Cylinder_34",
    "CamFollower/Cylinder_32",
    "CamFollower/CamFollowerTail",
    "CamFollowerHead/Cam_Follower_Head",
    "HummerHead/Hummer_Head_Right",
    "HummerHead/Hummer_Head_Left",
    "HummerHead/Cylinder_33",
    "WingLinkArcLeft/Wing_Link_Arc__2x__1",
    "WingLinkArcRight/Wing_Link_Arc__2x_",
    "WingLinkStraightRight/Wing_Link_Straight__2x_",
    "WingLinkStraightLeft/Wing_Link_Straight__2x__1",
    "WingRight/Shoulder_Thin__2x_",
    "ShoulderRight/Shoulder_Pivot__2x_",
    "WingRightConnector/Cylinder_34",
    "WingLeft/Shoulder_Thin__2x__mirrored",
    "ShoulderLeft/Shoulder_Pivot__2x__mirrored",
    "WingLeftConnector/Cylinder_35",
    "TailMount/Tail_Mount_Left",
    "TailMount/Tail_Mount_Right",
    "TailMount/Cylinder_32",
    "TailMount/Cylinder_33",
    "TailMount/Cylinder_34",
    "TailMount/Tail_Mount_Left_Standoff_A",
    "TailMount/Tail_Mount_Left_Standoff_B",
    "TailPinion/Tail_Pinion_Thick",
    "TailPinion/Tail_Pinion_Thin",
    "Frame/FrameMesh",
    "Frame/Cylinders/Cylinder",
    "Frame/Cylinders/Cylinder_01",
    "Frame/Cylinders/Cylinder_02",
    "Frame/Cylinders/Cylinder_03",
    "Frame/Cylinders/Cylinder_04",
    "Frame/Cylinders/Cylinder_05",
    "Frame/Cylinders/Cylinder_06",
    "Frame/Cylinders/Cylinder_07",
    "Frame/Cylinders/Cylinder_08",
    "Frame/Cylinders/Cylinder_09",
    "Frame/Cylinders/Cylinder_10",
    "Frame/Cylinders/Cylinder_11",
    "Frame/Cylinders/Cylinder_12",
    "Frame/Cylinders/Cylinder_13",
    "Frame/Cylinders/Cylinder_14",
    "Frame/Cylinders/Cylinder_15",
    "Frame/Cylinders/Cylinder_16",
    "Frame/Cylinders/Cylinder_17",
    "Frame/Cylinders/Cylinder_18",
    "Frame/Cylinders/Cylinder_19",
    "Frame/Cylinders/Cylinder_20",
    "Frame/Cylinders/Cylinder_21",
    "Frame/Cylinders/Cylinder_22",
    "Frame/Cylinders/Cylinder_23",
    "Frame/Cylinders/Cylinder_24",
    "Frame/Cylinders/Cylinder_25",
    "Frame/Cylinders/Cylinder_26",
    "Frame/Cylinders/Cylinder_27",
    "Frame/Cylinders/Cylinder_28",
    "Frame/Cylinders/Cylinder_29",
    "Frame/Support_Frame",
    "Frame/Cylinder",
    "FrameGround/Base",
    "FrameGround/Flower/Flower_Stem",
    "FrameGround/Flower/Slider/Cube",
]


HIDDEN_SHAPES = ["Plane", "FrameGround/Cylinder_33", "FrameGround/Cylinder_34", "FrameGround/Cylinder_35"]

SHAPE_COLORS = {
    "CamFollower/CamFollowerTail": (0.47471, 0.556962, 0.52269),
    "CamFollowerBody/Cam_Follower_Body": (0.47471, 0.556962, 0.52269),
    "CamFollowerHead/Cam_Follower_Head": (0.47471, 0.556962, 0.52269),
    "CamWheelBottom/Cam_Wheel_Bottom": (0.476793, 0.47277, 0.47277),
    "CamWheelHead/Cam_Wheel_Body": (0.476793, 0.47277, 0.47277),
    "CamWheelTail/Cam_Wheel_Tail": (0.476793, 0.47277, 0.47277),
    "Crank/Crank": (0.476793, 0.47277, 0.47277),
    "Crank/Gear_Small_Lower": (0.508555, 0.511331, 0.611814),
    "Frame/Cylinder": (1e-06, 1e-06, 1e-06),
    "Frame/Cylinder_01": (1e-06, 1e-06, 1e-06),
    "FrameGround/Base": (1e-06, 1e-06, 1e-06),
    "FrameGround/Flower/Flower_Center_01": (1.0, 0.835443, 0.0),
    "FrameGround/Flower/Flower_Petal/Flower_Petal_Front__5x__01": (1.0, 0.0, 0.0),
    "FrameGround/Flower/Flower_Petal/Flower_Petal_Rear__5x__01": (0.47471, 0.556962, 0.52269),
    "FrameGround/Flower/Flower_Petal_01/Flower_Petal_Front__5x__01": (1.0, 0.0, 0.0),
    "FrameGround/Flower/Flower_Petal_01/Flower_Petal_Rear__5x__01": (0.47471, 0.556962, 0.52269),
    "FrameGround/Flower/Flower_Petal_02/Flower_Petal_Front__5x__01": (1.0, 0.0, 0.0),
    "FrameGround/Flower/Flower_Petal_02/Flower_Petal_Rear__5x__01": (0.47471, 0.556962, 0.52269),
    "FrameGround/Flower/Flower_Petal_03/Flower_Petal_Front__5x__01": (1.0, 0.0, 0.0),
    "FrameGround/Flower/Flower_Petal_03/Flower_Petal_Rear__5x__01": (0.47471, 0.556962, 0.52269),
    "FrameGround/Flower/Flower_Petal_04/Flower_Petal_Front__5x__01": (1.0, 0.0, 0.0),
    "FrameGround/Flower/Flower_Petal_04/Flower_Petal_Rear__5x__01": (0.47471, 0.556962, 0.52269),
    "FrameGround/Flower/Slider/Cube": (0.47471, 0.556962, 0.52269),
    "FrameGround/Flower/Slider/Cylinder_44": (0.47471, 0.556962, 0.52269),
    "FrameGround/Flower/Slider/Cylinder_45": (0.47471, 0.556962, 0.52269),
    "FrameGround/Frame_Support_A": (0.47471, 0.556962, 0.52269),
    "FrameGround/Frame_Support_B": (0.47471, 0.556962, 0.52269),
    "Gear_Large__3x_00/Gear_Large__3x_0": (0.508555, 0.511331, 0.611814),
    "Gear_Large__3x_01/Gear_Large__3x_01": (0.508555, 0.511331, 0.611814),
    "Gear_Large__3x_02/Gear_Large__3x_02": (0.508555, 0.511331, 0.611814),
    "GearedSpinner/Gear_Small_Upper": (0.508555, 0.511331, 0.611814),
    "HummerBody/Hummer_Body_Left": (1e-06, 1e-06, 1e-06),
    "HummerBody/Hummer_Body_Right": (1e-06, 1e-06, 1e-06),
    "HummerBody/Tail_Fan_Gear": (0.508555, 0.511331, 0.611814),
    "HummerHead/Hummer_Bill": (1e-06, 1e-06, 1e-06),
    "HummerHead/Hummer_Head_Left": (0.058751, 0.0807, 0.232068),
    "HummerHead/Hummer_Head_Right": (0.058751, 0.0807, 0.232068),
    "Hypocycloid_Gear__3x_0/Hypocycloid_Gear__3x_0_mesh": (0.058751, 0.0807, 0.232068),
    "Hypocycloid_Gear__3x_01/Hypocycloid_Gear__3x_0_mesh": (0.058751, 0.0807, 0.232068),
    "Hypocycloid_Gear__3x_02/Hypocycloid_Gear__3x_0_mesh": (0.058751, 0.0807, 0.232068),
    "ShoulderLeft/Shoulder_Pivot__2x__mirrored": (1e-06, 1e-06, 1e-06),
    "ShoulderRight/Shoulder_Pivot__2x_": (1e-06, 1e-06, 1e-06),
    "Tail/Tail_Feather_A__2x_": (1e-06, 1e-06, 1e-06),
    "Tail/Tail_Feather_A__2x__mirrored": (1e-06, 1e-06, 1e-06),
    "Tail/Tail_Feather_B__2x_": (1e-06, 1e-06, 1e-06),
    "Tail/Tail_Feather_B__2x__mirrored": (1e-06, 1e-06, 1e-06),
    "Tail/Tail_Feather_C": (1e-06, 1e-06, 1e-06),
    "TailPinion/Tail_Pinion_Thick": (0.508555, 0.511331, 0.611814),
    "TailPinion/Tail_Pinion_Thin": (0.508555, 0.511331, 0.611814),
    "TailRack/Tail_Rack": (0.508555, 0.511331, 0.611814),
    "WingLeft/Wing__2x__mirrored": (1e-06, 1e-06, 1e-06),
    "WingLeftConnector/Cylinder_35": (0.476793, 0.47277, 0.47277),
    "WingLeftConnector/Sphere_01": (0.476793, 0.47277, 0.47277),
    "WingLinkArcLeft/Wing_Link_Arc__2x__1": (1e-06, 1e-06, 1e-06),
    "WingLinkArcRight/Wing_Link_Arc__2x_": (1e-06, 1e-06, 1e-06),
    "WingRight/Wing__2x_": (1e-06, 1e-06, 1e-06),
    "WingRightConnector/Cylinder_34": (0.476793, 0.47277, 0.47277),
    "WingRightConnector/Sphere": (0.476793, 0.47277, 0.47277),
}


# Authored physics material density (kg/m^3) and dynamic friction.
SHAPE_MATERIALS = {
    "CamFollower/CamFollowerTail": (600.0, 0.0),
    "CamFollowerBody/Cam_Follower_Body": (60000.003, 0.0),
    "CamFollowerHead/Cam_Follower_Head": (4000.0, 0.0),
    "CamWheelBottom/Cam_Wheel_Bottom": (4000.0, 0.0),
    "CamWheelHead/Cam_Wheel_Body": (4000.0, 0.0),
    "CamWheelTail/Cam_Wheel_Tail": (4000.0, 0.0),
    "Crank/Cylinder_40": (600.0, 0.0),
    "Frame/Cylinder": (5000.0, 0.0),
    "Frame/Cylinder_35": (600.0, 0.0),
    "Frame/Cylinder_36": (600.0, 0.0),
    "Frame/Cylinder_37": (600.0, 0.0),
    "Frame/Cylinder_38": (600.0, 0.0),
    "Frame/Cylinder_39": (600.0, 0.0),
    "Frame/Cylinder_40": (600.0, 0.0),
    "Frame/Cylinder_41": (600.0, 0.0),
    "Frame/Cylinder_42": (600.0, 0.0),
    "Frame/FrameMesh": (4000.0, 0.0),
    "Frame/Support_Frame": (4000.0, 0.0),
    "FrameGround/Base": (500.0, 0.5),
    "FrameGround/Cylinder_40": (600.0, 0.0),
    "FrameGround/Cylinder_41": (600.0, 0.0),
    "FrameGround/Cylinder_42": (600.0, 0.0),
    "FrameGround/Cylinder_43": (600.0, 0.0),
    "FrameGround/Cylinder_44": (600.0, 0.0),
    "FrameGround/Flower/Flower_Stem": (5000.0, 0.0),
    "FrameGround/Flower/Slider/Cube": (4000.0, 0.0),
    "FrameGround/Flower/Slider/Cylinder_44": (600.0, 0.0),
    "FrameGround/Flower/Slider/Cylinder_45": (600.0, 0.0),
    "Gear_Large__3x_00/Gear_Large__3x_0": (4000.0, 0.0),
    "Gear_Large__3x_01/Gear_Large__3x_01": (4000.0, 0.0),
    "Gear_Large__3x_02/Gear_Large__3x_02": (4000.0, 0.0),
    "GearedSpinner/Cylinder_31": (4000.0, 0.0),
    "GearedSpinner/Cylinder_32": (4000.0, 0.0),
    "GearedSpinner/Cylinder_33": (4000.0, 0.0),
    "GearedSpinner/Spinner_Link_Thick": (4000.0, 0.0),
    "GearedSpinner/Spinner_Link_Thin": (60000.003, 0.0),
    "HummerBody/Cylinder_30": (4000.0, 0.0),
    "HummerBody/Cylinder_43": (600.0, 0.0),
    "HummerBody/Cylinder_44": (600.0, 0.0),
    "HummerBody/Cylinder_45": (600.0, 0.0),
    "HummerBody/Hummer_Body_Left": (4000.0, 0.0),
    "HummerBody/Hummer_Body_Right": (4000.0, 0.0),
    "HummerHead/Cylinder_33": (4000.0, 0.0),
    "HummerHead/Hummer_Head_Left": (4000.0, 0.0),
    "HummerHead/Hummer_Head_Right": (4000.0, 0.0),
    "Hypocycloid_Gear__3x_0/Hypocycloid_Gear__3x_0_mesh": (4000.0, 0.0),
    "Hypocycloid_Gear__3x_01/Hypocycloid_Gear__3x_0_mesh": (4000.0, 0.0),
    "Hypocycloid_Gear__3x_02/Hypocycloid_Gear__3x_0_mesh": (4000.0, 0.0),
    "ShoulderLeft/Shoulder_Pivot__2x__mirrored": (4000.0, 0.0),
    "ShoulderRight/Shoulder_Pivot__2x_": (4000.0, 0.0),
    "Tail/Tail_Feather_A__2x_": (600.0, 0.0),
    "Tail/Tail_Feather_A__2x__mirrored": (600.0, 0.0),
    "Tail/Tail_Feather_B__2x_": (600.0, 0.0),
    "Tail/Tail_Feather_B__2x__mirrored": (600.0, 0.0),
    "Tail/Tail_Feather_C": (600.0, 0.0),
    "TailMount/Cylinder_34": (600.0, 0.0),
    "TailMount/Tail_Mount_Left": (4000.0, 0.0),
    "TailMount/Tail_Mount_Right": (4000.0, 0.0),
    "TailRack/Cylinder_34": (4000.0, 0.0),
    "WingLeft/Shoulder_Thin__2x__mirrored": (4000.0, 0.0),
    "WingLeftConnector/Cylinder_35": (60000.003, 0.0),
    "WingLinkArcLeft/Cylinder_32": (4000.0, 0.0),
    "WingLinkArcLeft/Cylinder_33": (4000.0, 0.0),
    "WingLinkArcLeft/Wing_Link_Arc__2x__1": (4000.0, 0.0),
    "WingLinkArcRight/Cylinder_31": (4000.0, 0.0),
    "WingLinkArcRight/Cylinder_32": (4000.0, 0.0),
    "WingLinkArcRight/Wing_Link_Arc__2x_": (4000.0, 0.0),
    "WingLinkStraightLeft/Wing_Link_Straight__2x__1": (4000.0, 0.0),
    "WingLinkStraightRight/Wing_Link_Straight__2x_": (4000.0, 0.0),
    "WingRight/Shoulder_Thin__2x_": (4000.0, 0.0),
    "WingRight/Wing__2x_": (4000.0, 0.0),
    "WingRightConnector/Cylinder_34": (60000.003, 0.0),
}


def _transform(values):
    return wp.transform(values[:3], values[3:])


# Authored per-collider SDF grid resolutions from the source USD.
SHAPE_SDF_RESOLUTIONS = {
    "CamFollower/CamFollowerTail": 400,
    "CamFollower/Cylinder_32": 64,
    "CamFollowerBody/Cam_Follower_Body": 200,
    "CamFollowerBody/Cylinder_31": 64,
    "CamFollowerHead/Cam_Follower_Head": 200,
    "CamWheelBottom/Cam_Wheel_Bottom": 256,
    "CamWheelBottom/Cylinder_38": 64,
    "CamWheelBottom/Cylinder_39": 64,
    "CamWheelBottom/Cylinder_40": 64,
    "CamWheelBottom/Cylinder_41": 64,
    "CamWheelHead/Cam_Wheel_Body": 256,
    "CamWheelHead/Cam_Wheel_Head": 256,
    "CamWheelHead/Cylinder_30": 64,
    "CamWheelHead/Cylinder_31": 64,
    "CamWheelHead/Cylinder_32": 64,
    "CamWheelHead/Cylinder_33": 64,
    "CamWheelTail/Cam_Wheel_Tail": 256,
    "CamWheelTail/Cylinder_34": 64,
    "CamWheelTail/Cylinder_35": 64,
    "CamWheelTail/Cylinder_36": 64,
    "CamWheelTail/Cylinder_37": 64,
    "Crank/Crank": 128,
    "Crank/Gear_Small_Lower": 128,
    "Frame/Cylinder": 64,
    "Frame/Cylinders/Cylinder": 64,
    "Frame/Cylinders/Cylinder_01": 64,
    "Frame/Cylinders/Cylinder_02": 64,
    "Frame/Cylinders/Cylinder_03": 64,
    "Frame/Cylinders/Cylinder_04": 64,
    "Frame/Cylinders/Cylinder_05": 64,
    "Frame/Cylinders/Cylinder_06": 64,
    "Frame/Cylinders/Cylinder_07": 64,
    "Frame/Cylinders/Cylinder_08": 64,
    "Frame/Cylinders/Cylinder_09": 64,
    "Frame/Cylinders/Cylinder_10": 64,
    "Frame/Cylinders/Cylinder_11": 64,
    "Frame/Cylinders/Cylinder_12": 64,
    "Frame/Cylinders/Cylinder_13": 64,
    "Frame/Cylinders/Cylinder_14": 64,
    "Frame/Cylinders/Cylinder_15": 64,
    "Frame/Cylinders/Cylinder_16": 64,
    "Frame/Cylinders/Cylinder_17": 64,
    "Frame/Cylinders/Cylinder_18": 64,
    "Frame/Cylinders/Cylinder_19": 64,
    "Frame/Cylinders/Cylinder_20": 64,
    "Frame/Cylinders/Cylinder_21": 64,
    "Frame/Cylinders/Cylinder_22": 64,
    "Frame/Cylinders/Cylinder_23": 64,
    "Frame/Cylinders/Cylinder_24": 64,
    "Frame/Cylinders/Cylinder_25": 64,
    "Frame/Cylinders/Cylinder_26": 64,
    "Frame/Cylinders/Cylinder_27": 64,
    "Frame/Cylinders/Cylinder_28": 64,
    "Frame/Cylinders/Cylinder_29": 64,
    "Frame/FrameMesh": 400,
    "Frame/Support_Frame": 400,
    "FrameGround/Base": 128,
    "FrameGround/Flower/Flower_Stem": 200,
    "FrameGround/Flower/Slider/Cube": 100,
    "Gear_Large__3x_00/Gear_Large__3x_0": 256,
    "Gear_Large__3x_01/Gear_Large__3x_01": 256,
    "Gear_Large__3x_02/Gear_Large__3x_02": 256,
    "GearedSpinner/Gear_Small_Upper": 128,
    "GearedSpinner/Spinner_Link_Thick": 64,
    "GearedSpinner/Spinner_Link_Thin": 64,
    "HummerBody/Cylinder_30": 64,
    "HummerBody/Hummer_Body_Left": 128,
    "HummerBody/Hummer_Body_Right": 128,
    "HummerBody/Tail_Fan_Gear": 128,
    "HummerHead/Cylinder_33": 64,
    "HummerHead/Hummer_Head_Left": 128,
    "HummerHead/Hummer_Head_Right": 128,
    "Hypocycloid_Gear__3x_0/Hypocycloid_Gear__3x_0_mesh": 256,
    "Hypocycloid_Gear__3x_01/Hypocycloid_Gear__3x_0_mesh": 256,
    "Hypocycloid_Gear__3x_02/Hypocycloid_Gear__3x_0_mesh": 256,
    "ShoulderLeft/Shoulder_Pivot__2x__mirrored": 100,
    "ShoulderRight/Shoulder_Pivot__2x_": 100,
    "Tail/Tail_Feather_A__2x_": 300,
    "Tail/Tail_Feather_A__2x__mirrored": 300,
    "Tail/Tail_Feather_B__2x_": 300,
    "Tail/Tail_Feather_B__2x__mirrored": 300,
    "Tail/Tail_Feather_C": 300,
    "TailMount/Cylinder_32": 64,
    "TailMount/Cylinder_33": 64,
    "TailMount/Cylinder_34": 64,
    "TailMount/Tail_Mount_Left": 100,
    "TailMount/Tail_Mount_Left_Standoff_A": 64,
    "TailMount/Tail_Mount_Left_Standoff_B": 64,
    "TailMount/Tail_Mount_Right": 100,
    "TailPinion/Tail_Pinion_Thick": 128,
    "TailPinion/Tail_Pinion_Thin": 128,
    "TailRack/Cylinder_34": 64,
    "TailRack/Tail_Rack": 256,
    "WingLeft/Shoulder_Thin__2x__mirrored": 64,
    "WingLeftConnector/Cylinder_35": 64,
    "WingLinkArcLeft/Wing_Link_Arc__2x__1": 64,
    "WingLinkArcRight/Wing_Link_Arc__2x_": 64,
    "WingLinkStraightLeft/Wing_Link_Straight__2x__1": 64,
    "WingLinkStraightRight/Wing_Link_Straight__2x_": 64,
    "WingRight/Shoulder_Thin__2x_": 64,
    "WingRightConnector/Cylinder_34": 64,
}

SHAPE_CONTACT_OFFSETS = {
    "Hypocycloid_Gear__3x_0/Hypocycloid_Gear__3x_0_mesh": 0.002,
    "Hypocycloid_Gear__3x_02/Hypocycloid_Gear__3x_0_mesh": 0.002,
    "Hypocycloid_Gear__3x_01/Hypocycloid_Gear__3x_0_mesh": 0.002,
    "HummerBody/Cylinder_30": 0.001,
    "CamFollowerBody/Cylinder_31": 0.001,
    "CamWheelHead/Cylinder_30": 0.001,
    "CamWheelHead/Cylinder_31": 0.001,
    "CamWheelHead/Cylinder_32": 0.001,
    "CamWheelHead/Cylinder_33": 0.001,
    "CamWheelBottom/Cylinder_38": 0.001,
    "CamWheelBottom/Cylinder_39": 0.001,
    "CamWheelBottom/Cylinder_40": 0.001,
    "CamWheelBottom/Cylinder_41": 0.001,
    "CamWheelTail/Cylinder_34": 0.001,
    "CamWheelTail/Cylinder_35": 0.001,
    "CamWheelTail/Cylinder_36": 0.001,
    "CamWheelTail/Cylinder_37": 0.001,
    "TailRack/Cylinder_34": 0.001,
    "CamFollower/Cylinder_32": 0.001,
    "HummerHead/Cylinder_33": 0.001,
    "WingRightConnector/Cylinder_34": 0.001,
    "WingLeftConnector/Cylinder_35": 0.001,
    "TailMount/Cylinder_32": 0.001,
    "TailMount/Cylinder_33": 0.001,
    "TailMount/Cylinder_34": 0.001,
    "Frame/Cylinders/Cylinder": 0.001,
    "Frame/Cylinders/Cylinder_01": 0.001,
    "Frame/Cylinders/Cylinder_02": 0.001,
    "Frame/Cylinders/Cylinder_03": 0.001,
    "Frame/Cylinders/Cylinder_04": 0.001,
    "Frame/Cylinders/Cylinder_05": 0.001,
    "Frame/Cylinders/Cylinder_06": 0.001,
    "Frame/Cylinders/Cylinder_07": 0.001,
    "Frame/Cylinders/Cylinder_08": 0.001,
    "Frame/Cylinders/Cylinder_09": 0.001,
    "Frame/Cylinders/Cylinder_10": 0.001,
    "Frame/Cylinders/Cylinder_11": 0.001,
    "Frame/Cylinders/Cylinder_12": 0.001,
    "Frame/Cylinders/Cylinder_13": 0.001,
    "Frame/Cylinders/Cylinder_14": 0.001,
    "Frame/Cylinders/Cylinder_15": 0.001,
    "Frame/Cylinders/Cylinder_16": 0.001,
    "Frame/Cylinders/Cylinder_17": 0.001,
    "Frame/Cylinders/Cylinder_18": 0.001,
    "Frame/Cylinders/Cylinder_19": 0.001,
    "Frame/Cylinders/Cylinder_20": 0.001,
    "Frame/Cylinders/Cylinder_21": 0.001,
    "Frame/Cylinders/Cylinder_22": 0.001,
    "Frame/Cylinders/Cylinder_23": 0.001,
    "Frame/Cylinders/Cylinder_24": 0.001,
    "Frame/Cylinders/Cylinder_25": 0.001,
    "Frame/Cylinders/Cylinder_26": 0.001,
    "Frame/Cylinders/Cylinder_27": 0.001,
    "Frame/Cylinders/Cylinder_28": 0.001,
    "Frame/Cylinders/Cylinder_29": 0.001,
}


@wp.kernel
def _apply_source_damping(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_mass: wp.array[float],
    body_inv_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    rack: int,
    undamped_gear: int,
    body_f: wp.array[wp.spatial_vector],
):
    body = wp.tid()
    if body_inv_mass[body] == 0.0:
        return
    velocity = body_qd[body]
    linear = wp.vec3(velocity[0], velocity[1], velocity[2])
    angular = wp.vec3(velocity[3], velocity[4], velocity[5])
    force = wp.vec3(0.0)
    if body == rack:
        force = -body_mass[body] * linear
    torque = wp.vec3(0.0)
    if body != undamped_gear:
        rotation = wp.transform_get_rotation(body_q[body])
        local_angular = wp.quat_rotate_inv(rotation, angular)
        torque = -0.05 * wp.quat_rotate(rotation, body_inertia[body] * local_angular)
    body_f[body] += wp.spatial_vector(force, torque)


def apply_source_damping(model, state):
    """Apply the USD body damping rates in reciprocal seconds as drag forces."""
    rack = model.body_label.index("TailRack") if "TailRack" in model.body_label else -1
    gear = model.body_label.index("Gear_Large__3x_02") if "Gear_Large__3x_02" in model.body_label else -1
    wp.launch(
        _apply_source_damping,
        dim=model.body_count,
        inputs=[state.body_q, state.body_qd, model.body_mass, model.body_inv_mass, model.body_inertia, rack, gear],
        outputs=[state.body_f],
        device=model.device,
    )


def build_scene(
    *,
    body_count: int | None = None,
    fix_base: bool = False,
    contact_gap: float = 0.001,
    source_contact_offsets: bool = True,
    mesh_cylinders: bool = False,
    sdf_resolution: int = 0,
    counterweight_density_scale: float = 1.0,
    attach_flower_to_base: bool = False,
    enable_frame_drive: bool = True,
):
    """Build a connected prefix with optional counterweight and flower changes.

    ``counterweight_density_scale`` multiplies only Frame/Cylinder's authored
    density before accumulating body mass properties. ``attach_flower_to_base``
    merges the flower/slider shapes into FrameGround; the default preserves the
    source's separate kinematic flower.
    ``enable_frame_drive`` retains the source's 20-degree base/frame position
    drive; disable it to let the axle rotate without its spring and damper.
    """
    if not np.isfinite(counterweight_density_scale) or counterweight_density_scale < 0.0:
        raise ValueError("counterweight_density_scale must be finite and nonnegative")
    if body_count is None:
        body_count = len(BODY_ORDER)

    if not 1 <= body_count <= len(BODY_ORDER):
        raise ValueError(f"body_count must be between 1 and {len(BODY_ORDER)}")
    # Source gravity is 100 cm/s^2, expressed here in meters.
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -1.0))
    newton.solvers.SolverKamino.register_custom_attributes(builder)
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.mu = 0.5
    builder.default_shape_cfg.margin = 0.0
    if contact_gap < 0.0:
        raise ValueError("contact_gap must be nonnegative")
    builder.default_shape_cfg.gap = contact_gap
    selected = BODY_ORDER[:body_count]
    bodies = {name: builder.add_link(xform=_transform(BODY_POSES[name]), label=name) for name in selected}
    # The source flower is kinematic with a disabled base joint. Its geometry
    # already uses FrameGround coordinates, so a physical attachment can use
    # the base body directly, including all flower/slider mass contributions.
    flower = bodies["FrameGround"]
    if not attach_flower_to_base:
        flower = builder.add_link(xform=_transform(BODY_POSES["FrameGround"]), label="Flower", is_kinematic=True)
    assets = Path(newton.examples.get_asset_directory()) / "colibri"
    for name, kind, label, data, dimensions in SHAPES:
        if name not in bodies:
            continue
        body = flower if label.startswith("FrameGround/Flower/") else bodies[name]
        cfg = builder.default_shape_cfg.copy()
        cfg.has_shape_collision = label in COLLISION_LABELS
        if source_contact_offsets:
            cfg.gap = SHAPE_CONTACT_OFFSETS.get(label, cfg.gap)
        shape_sdf_resolution = sdf_resolution or SHAPE_SDF_RESOLUTIONS.get(label, 128)
        cfg.is_visible = label not in HIDDEN_SHAPES
        cfg.density, cfg.mu = SHAPE_MATERIALS.get(label, (1000.0, 0.5))
        if label == "Frame/Cylinder":
            cfg.density *= counterweight_density_scale
        if not cfg.has_shape_collision:
            cfg.density = 0.0
        # Unbound meshes in the source USD have white displayColor primvars.
        color = SHAPE_COLORS.get(label, (1.0, 1.0, 1.0))
        if kind == "cylinder":
            if mesh_cylinders:
                source = trimesh.creation.cylinder(radius=dimensions[0], height=2.0 * dimensions[1], sections=32)
                mesh = newton.Mesh(
                    np.asarray(source.vertices, dtype=np.float32),
                    np.asarray(source.faces, dtype=np.int32).flatten(),
                )
                if cfg.has_shape_collision:
                    mesh.build_sdf(
                        max_resolution=shape_sdf_resolution,
                        margin=max(0.0002, cfg.gap),
                        cache_dir=str(Path(tempfile.gettempdir()) / "newton_colibri_sdf_cache"),
                    )
                builder.add_shape_mesh(
                    body,
                    mesh=mesh,
                    xform=_transform(data),
                    cfg=cfg,
                    color=SHAPE_COLORS.get(label, (0.45, 0.48, 0.52)),
                    label=label,
                )
            else:
                builder.add_shape_cylinder(
                    body,
                    xform=_transform(data),
                    radius=dimensions[0],
                    half_height=dimensions[1],
                    cfg=cfg,
                    color=SHAPE_COLORS.get(label, (0.45, 0.48, 0.52)),
                    label=label,
                )
        else:
            source = trimesh.load(assets / data, force="mesh", process=False)
            affine = np.asarray(dimensions)
            vertices = np.asarray(source.vertices) @ affine[:, :3].T + affine[:, 3]
            center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
            mesh = newton.Mesh(
                (vertices - center).astype(np.float32), np.asarray(source.faces, dtype=np.int32).flatten()
            )
            if cfg.has_shape_collision:
                mesh.build_sdf(
                    max_resolution=shape_sdf_resolution,
                    margin=max(0.0002, cfg.gap),
                    cache_dir=str(Path(tempfile.gettempdir()) / "newton_colibri_sdf_cache"),
                )
            builder.add_shape_mesh(
                body,
                mesh=mesh,
                xform=wp.transform(center, wp.quat_identity()),
                cfg=cfg,
                color=color,
                label=label,
            )

    if fix_base:
        root_joint = builder.add_joint_fixed(
            -1, bodies["FrameGround"], parent_xform=_transform(BODY_POSES["FrameGround"]), label="GroundAnchor"
        )
    else:
        root_joint = builder.add_joint_free(bodies["FrameGround"], label="FrameGround/free")
    tree = [root_joint]
    connected = {"FrameGround"}
    pending = [j for j in JOINTS if j[0] in bodies and j[1] in bodies]
    # Keep loop closures outside the spanning-tree articulation.
    while pending:
        ready = next((j for j in pending if j[0] in connected and j[1] not in connected and j[2] == "revolute"), None)
        if ready is None:
            break
        a, b, kind, axis, frame_a, frame_b, drive = ready
        if not enable_frame_drive and (a, b) == ("FrameGround", "Frame"):
            drive = {}
        kwargs = dict(drive)
        if drive:
            # USD angular drives have no maximum-force limit.
            kwargs["effort_limit"] = float("inf")
            kwargs["actuator_mode"] = (
                newton.JointTargetMode.POSITION if drive.get("target_ke", 0) else newton.JointTargetMode.VELOCITY
            )
        index = builder.add_joint_revolute(
            bodies[a],
            bodies[b],
            axis=newton.Axis[axis],
            parent_xform=_transform(frame_a),
            child_xform=_transform(frame_b),
            label=f"{a}/{b}",
            **kwargs,
        )
        tree.append(index)
        connected.add(b)
        pending.remove(ready)
    builder.add_articulation(tree, label="Colibri")
    for a, b, kind, axis, frame_a, frame_b, _drive in pending:
        method = builder.add_joint_ball if kind == "ball" else builder.add_joint_revolute
        kwargs = {} if kind == "ball" else {"axis": newton.Axis[axis]}
        method(
            bodies[a],
            bodies[b],
            parent_xform=_transform(frame_a),
            child_xform=_transform(frame_b),
            label=f"{a}/{b}/closure",
            **kwargs,
        )
    if "TailRack" in bodies:
        joint = builder.add_joint_free(bodies["TailRack"], label="TailRack/free")
        builder.add_articulation([joint], label="TailRack")
    if not attach_flower_to_base:
        flower_joint = builder.add_joint_free(flower, label="Flower/free")
        builder.add_articulation([flower_joint], label="Flower")
    builder.add_ground_plane(height=0.002965275)
    return builder


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = getattr(args, "substeps", 20)
        if self.sim_substeps < 1:
            raise ValueError("substeps must be positive")
        self.contact_updates = getattr(args, "contact_updates_per_frame", 2)
        if self.contact_updates < 0 or (self.contact_updates and self.sim_substeps % self.contact_updates):
            raise ValueError("contact updates must be zero or divide the physics substeps")
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.fix_base = getattr(args, "fix_base", False)
        self.source_damping = getattr(args, "source_damping", False)
        builder = build_scene(
            body_count=args.body_count,
            fix_base=self.fix_base,
            contact_gap=getattr(args, "contact_gap", 0.001),
            source_contact_offsets=getattr(args, "source_contact_offsets", True),
            mesh_cylinders=getattr(args, "mesh_cylinders", False),
            sdf_resolution=getattr(args, "sdf_resolution", 0),
        )
        self.model = builder.finalize(skip_validation_joints=True)
        config = newton.solvers.SolverKamino.Config.from_model(
            self.model,
            dynamics_solver="dvi",
            sparse_dynamics=True,
            sparse_jacobian=True,
        )
        config.constraints.alpha = 0.1
        config.use_collision_detector = self.contact_updates == 0
        if self.contact_updates:
            # Keep separated cached contacts so they can activate before the next refresh.
            config.dynamics.cull_speculative_contacts = False
        config.dvi.use_schur_complement = True
        config.dvi.bilateral_solver_type = "LLTBRCM"
        config.dvi.omega = 1.0
        config.dvi.max_alternating_iterations = getattr(args, "iterations", 16)
        # Keep the bilateral response fixed across sweeps to enable compact Schur.
        if config.dvi.max_alternating_iterations < 1:
            raise ValueError("iterations must be positive")
        config.dvi.bilateral_solve_interval = config.dvi.max_alternating_iterations
        config.dvi.tolerance = 1.0e-5
        self.contact_capacity = getattr(args, "contact_capacity", 0) or (
            8192 if getattr(args, "mesh_cylinders", False) else 1024
        )
        config.collision_detector.max_contacts_per_world = self.contact_capacity
        if self.contact_updates:
            self.model.rigid_contact_max = self.contact_capacity * self.model.world_count
        self.solver = newton.solvers.SolverKamino(self.model, config=config)
        self.collision_pipeline = None
        self.contacts = None
        if self.contact_updates:
            self.collision_pipeline = newton.CollisionPipeline(
                self.model, rigid_contact_max=self.model.rigid_contact_max, contact_matching="sticky"
            )
            self.contacts = self.collision_pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.solver.reset(self.state_0)
        self.initial_q = self.state_0.body_q.numpy().copy()
        self.viewer.set_model(self.model)
        if hasattr(viewer, "set_camera"):
            viewer.set_camera(wp.vec3(0.4, -0.7, 0.35), pitch=-10, yaw=120)
        self.graph = None
        # An even number of state swaps returns the graph to its input buffer.
        if self.model.device.is_cuda and self.sim_substeps % 2 == 0:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for substep in range(self.sim_substeps):
            if self.collision_pipeline is not None and substep % (self.sim_substeps // self.contact_updates) == 0:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.state_0.clear_forces()
            if self.source_damping:
                apply_source_damping(self.model, self.state_0)
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        """Check finite motion and joint attachment errors."""
        q = self.state_0.body_q.numpy()
        qd = self.state_0.body_qd.numpy()
        assert np.isfinite(q).all() and np.isfinite(qd).all(), "Non-finite body state"
        if getattr(self, "fix_base", True):
            displacement = q[:, :3] - self.initial_q[:, :3]
        else:
            # Global sliding/yaw does not detach a body from this free assembly.
            # This bound tests attachment, not the accuracy of support friction.
            base = self.model.body_label.index("FrameGround")
            relative = []
            for poses in (q, self.initial_q):
                vectors = poses[:, :3] - poses[base, :3]
                inverse_xyz = -poses[base, 3:6]
                relative.append(
                    vectors + 2.0 * np.cross(inverse_xyz, np.cross(inverse_xyz, vectors) + poses[base, 6] * vectors)
                )
            assembly = np.array([name != "Flower" for name in self.model.body_label])
            displacement = (relative[0] - relative[1])[assembly]
        assert np.max(np.linalg.norm(displacement, axis=1)) < 0.5, "Body escaped assembly"
        assert np.max(np.linalg.norm(qd[:, :3], axis=1)) < 10.0, "Excessive linear velocity"
        assert np.max(np.linalg.norm(qd[:, 3:], axis=1)) < 100.0, "Excessive angular velocity"
        if getattr(self, "fix_base", True):
            np.testing.assert_allclose(q[0], self.initial_q[0], atol=1.0e-5, err_msg="Base moved")
        if "TailRack" in self.model.body_label:
            rack = self.model.body_label.index("TailRack")
            mount = self.model.body_label.index("TailMount")
            separation = np.linalg.norm(q[rack, :3] - q[mount, :3])
            assert separation < 0.05, f"Tail rack lost its contact support: {separation:.4f} m"

        parents = self.model.joint_parent.numpy()
        children = self.model.joint_child.numpy()
        frames_a = self.model.joint_X_p.numpy()
        frames_b = self.model.joint_X_c.numpy()
        types = self.model.joint_type.numpy()
        axes = self.model.joint_axis.numpy()
        dof_starts = self.model.joint_qd_start.numpy()
        for j, (a, b) in enumerate(zip(parents, children, strict=True)):
            if types[j] == newton.JointType.FREE:
                continue
            parent = wp.transform_identity() if a < 0 else _transform(q[a])
            anchor_a = wp.transform_point(parent, wp.vec3(frames_a[j, :3]))
            anchor_b = wp.transform_point(_transform(q[b]), wp.vec3(frames_b[j, :3]))
            error = np.linalg.norm(np.asarray(anchor_a) - np.asarray(anchor_b))
            assert error < 0.002, f"Joint {self.model.joint_label[j]} separated by {error:.6f} m"
            if types[j] == newton.JointType.REVOLUTE:
                axis = wp.vec3(axes[dof_starts[j]])
                rotation_a = wp.mul(parent.q, wp.quat(frames_a[j, 3:]))
                rotation_b = wp.mul(wp.quat(q[b, 3:]), wp.quat(frames_b[j, 3:]))
                alignment = np.linalg.norm(
                    np.asarray(wp.quat_rotate(rotation_a, axis)) - np.asarray(wp.quat_rotate(rotation_b, axis))
                )
                assert alignment < 0.03, f"Joint {self.model.joint_label[j]} axes misaligned: {alignment:.6f}"

    def test_final(self):
        """Check the final mechanism state."""
        self.test_post_step()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--body-count", type=int, default=len(BODY_ORDER), help="Build the first N bodies, starting at FrameGround."
        )
        parser.add_argument(
            "--contact-updates-per-frame",
            type=int,
            default=2,
            help="Contact refreshes per 60 Hz frame; 2 means 120 Hz, 0 updates each physics substep.",
        )
        parser.add_argument(
            "--contact-capacity",
            type=int,
            default=0,
            help="Contact capacity per world; 0 selects a scene-dependent capacity.",
        )
        parser.add_argument(
            "--sdf-resolution",
            type=int,
            default=0,
            help="Maximum SDF grid resolution; 0 uses authored per-shape resolutions.",
        )
        parser.add_argument(
            "--mesh-cylinders",
            action="store_true",
            help="Use 32-sided SDF cylinder meshes for source collision comparisons.",
        )
        parser.add_argument("--source-damping", action="store_true", help="Apply the authored body damping rates.")
        parser.add_argument(
            "--source-contact-offsets",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Apply authored contact gaps of 0.001 and 0.002 meters.",
        )
        parser.add_argument("--contact-gap", type=float, default=0.001, help="Contact detection gap in meters.")
        parser.add_argument("--fix-base", action="store_true", help="Anchor the base for mechanism diagnostics.")
        parser.add_argument("--substeps", type=int, default=20, help="Physics substeps per 60 Hz frame.")
        parser.add_argument("--iterations", type=int, default=16, help="DVI alternating iterations per substep.")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
