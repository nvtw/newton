# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device port of the source CC01 case line and TS01 sealer PLCs, one thread per packer cell.

A line-by-line port of ``Controller.step`` (``case_conveyor_cc01/controller.py``),
``CaseLine.step`` and ``Sealer.step`` (``packer_l02/lines.py``) and ``CaseLineL02.step``
(``packer_l02/lines_l02.py``), as the plant runs them: no drag-test cases, no
preloaded boxes, no robot stand-in timer and no early feed. Box identities are
plant case indices; event logs are not kept. Times are float64, as in the source.
"""

import numpy as np
import warp as wp

# Station states of the CC01 lock station, in the source controller's names.
CASE_STATIONS = ("EMPTY", "FEEDING", "SQUARING", "LOCKING", "LOCKED", "RELEASING", "FAULT")
SEALER_STATES = ("EMPTY", "RECEIVE", "PROCESS", "SEALED", "RELEASE", "FAULT")
# Zones Z1..Z8 and sealer zones S1, S2; actuators PreStop, RearBlade, FrontStop, Clamp, EndStop, SwapStop.
ZONE_NAMES = ("Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8", "S1", "S2")
ACTUATOR_NAMES = ("PreStop", "RearBlade", "FrontStop", "Clamp", "EndStop", "SwapStop")

# Sensor columns: PE_Z1..PE_Z8 (zone i uses sensor i), PE_TS_IN, PE_TS_SWAP, PE_TS_OUT, SPAWN.
_TS_SWAP = wp.constant(9)
_TS_OUT = wp.constant(10)
_SPAWN = wp.constant(11)
_PRE_STOP = wp.constant(0)
_REAR_BLADE = wp.constant(1)
_FRONT_STOP = wp.constant(2)
_CLAMP = wp.constant(3)
_END_STOP = wp.constant(4)


@wp.struct
class CaseLines:
    """Per-cell state of the case line and sealer PLCs, and their constants."""

    t: wp.array[wp.float64]
    station: wp.array[int]
    t_state: wp.array[wp.float64]
    ready: wp.array[int]
    picks_done: wp.array[int]
    acknowledged: wp.array[int]
    occ: wp.array2d[int]
    hold: wp.array2d[int]
    tail: wp.array2d[wp.float64]
    seat_since: wp.array2d[wp.float64]  # NaN when unset
    prev: wp.array2d[int]
    act: wp.array2d[wp.float64]
    prestop_up_at: wp.array[wp.float64]  # NaN when unset
    front_up_at: wp.array[wp.float64]
    release8: wp.array[wp.float64]
    end_up_at: wp.array[wp.float64]
    magazine: wp.array[int]
    last_spawn: wp.array[wp.float64]
    supply: wp.array2d[int]
    supplied: wp.array[int]
    sealer: wp.array[int]
    sealer_t_state: wp.array[wp.float64]
    sealer_box: wp.array[int]
    sealer_stop: wp.array[wp.float64]
    sealer_stop_up_at: wp.array[wp.float64]
    sealer_seat_since: wp.array[wp.float64]
    swap_request: wp.array[int]
    zone_velocity: wp.array2d[wp.float64]
    status: wp.array2d[int]
    dt: wp.float64
    v_tr: wp.float64
    v_st: wp.float64
    ramp_up: wp.float64
    ramp_down: wp.float64
    square: wp.float64
    blade_move: wp.float64
    clamp_move: wp.float64
    clamp_settle: wp.float64
    front_clear: wp.float64
    prestop_clear: wp.float64
    endzone_settle: wp.float64
    spawn_interval: wp.float64
    clamp_ok_window: float
    clamp_on_box: float
    up: wp.float64
    down: wp.float64
    clamp_open: wp.float64
    clamp_closed: wp.float64
    stop_up: wp.float64
    stop_down: wp.float64
    process: wp.float64


@wp.func
def _transfer(lines: CaseLines, cell: int, t: wp.float64, src: int, dst: int):
    lines.occ[cell, dst] = lines.occ[cell, src]
    lines.occ[cell, src] = -1
    lines.hold[cell, src] = 0
    lines.tail[cell, src] = t + wp.float64(0.6)


@wp.func
def _set_station(lines: CaseLines, cell: int, t: wp.float64, state: int):
    lines.station[cell] = state
    lines.t_state[cell] = t


@wp.func
def _set_sealer(lines: CaseLines, cell: int, t: wp.float64, state: int):
    lines.sealer[cell] = state
    lines.sealer_t_state[cell] = t


@wp.func
def station_offers_case(ready: int, station: int, picks_done: int) -> bool:
    """Whether the KR10 may place a row into the case locked at the pack station.

    A filled case stays locked until the sealer takes it. The source KR10
    (``robots_l02._box_ok``) refuses a case it already filled; without that, a
    backed-up outfeed made it place a fourth row onto the full case.
    """
    return ready != 0 and station == 4 and picks_done < 3


@wp.kernel
def case_line_plcs(
    lines: CaseLines,
    hits: wp.array2d[int],
    observation: wp.array2d[float],
    sealed: wp.array[int],
    outfeed_ok: wp.array[int],
    case_bodies: wp.array2d[int],
    case_local: wp.array2d[wp.transform],
    speed: wp.array2d[wp.float64],
    closed: wp.array[int],
    spawn_enabled: wp.array[int],
    spawn_bodies: wp.array2d[int],
    spawn_local: wp.array2d[wp.transform],
    line_ready: wp.array[int],
):
    """One scan of a cell's case line, sealer, zone ramps and case supply."""
    cell = wp.tid()
    t = lines.t[cell] + lines.dt
    lines.t[cell] = t
    v_tr = lines.v_tr
    v_st = lines.v_st

    # The KR10 reports three placements per filled case (robot_done).
    completed = int(observation[cell, 13]) // 12
    if completed > lines.acknowledged[cell]:
        if lines.station[cell] == 4 and lines.picks_done[cell] < 3:
            lines.picks_done[cell] = 3
        lines.acknowledged[cell] = completed

    sealer_ready = (
        lines.sealer[cell] == 0 and lines.sealer_stop[cell] == lines.stop_up and wp.isnan(lines.sealer_stop_up_at[cell])
    )

    # ---- CaseLine.step / Controller.step ----------------------------------------------------
    fell8 = lines.prev[cell, 7] != 0 and hits[cell, 7] == 0
    for z in range(10):
        speed[cell, z] = wp.float64(0.0)

    # Outfeed Z4..Z8: zero-pressure accumulation, evaluated downstream first.
    for k in range(5):
        i = 7 - k
        beam = hits[cell, i] != 0
        fell = lines.prev[cell, i] != 0 and not beam
        if lines.occ[cell, i] < 0:
            lines.hold[cell, i] = 0
            if t < lines.tail[cell, i]:
                speed[cell, i] = v_tr
            continue
        if i == 7:
            if beam:
                if wp.isnan(lines.seat_since[cell, i]):
                    lines.seat_since[cell, i] = t
                if t - lines.seat_since[cell, i] >= lines.endzone_settle:
                    lines.hold[cell, i] = 1
            else:
                lines.seat_since[cell, i] = wp.float64(wp.nan)
            if lines.hold[cell, i] != 0:
                speed[cell, i] = wp.float64(0.0)
            else:
                speed[cell, i] = v_tr
            continue
        if beam and lines.occ[cell, i + 1] >= 0:
            lines.hold[cell, i] = 1
        else:
            lines.hold[cell, i] = 0
            speed[cell, i] = v_tr
            if fell and lines.occ[cell, i + 1] < 0:
                _transfer(lines, cell, t, i, i + 1)
        if lines.hold[cell, i] != 0:
            speed[cell, i] = wp.float64(0.0)
    # A zone receiving a box runs while the box crosses in.
    for i in range(4, 8):
        if speed[cell, i - 1] > wp.float64(0.0) and lines.occ[cell, i - 1] >= 0 and lines.occ[cell, i] < 0:
            speed[cell, i] = wp.max(speed[cell, i], v_tr)

    # Lock station Z3.
    station = lines.station[cell]
    age = t - lines.t_state[cell]
    if station == 0:  # EMPTY
        lines.act[cell, _REAR_BLADE] = lines.down
        lines.act[cell, _CLAMP] = lines.clamp_open
        if not wp.isnan(lines.front_up_at[cell]) and t >= lines.front_up_at[cell]:
            lines.act[cell, _FRONT_STOP] = lines.up
            lines.front_up_at[cell] = wp.float64(wp.nan)
        if (
            lines.act[cell, _FRONT_STOP] == lines.up
            and wp.isnan(lines.front_up_at[cell])
            and lines.occ[cell, 2] < 0
            and lines.occ[cell, 1] >= 0
            and lines.hold[cell, 1] != 0
        ):
            lines.act[cell, _PRE_STOP] = lines.down
            _set_station(lines, cell, t, 1)
    elif station == 1:  # FEEDING
        if age >= lines.blade_move:
            speed[cell, 1] = v_st
            speed[cell, 2] = v_st
        if lines.prev[cell, 1] != 0 and hits[cell, 1] == 0 and lines.occ[cell, 1] >= 0:
            _transfer(lines, cell, t, 1, 2)
            lines.prestop_up_at[cell] = t + lines.prestop_clear
        if hits[cell, 2] != 0 and lines.occ[cell, 2] >= 0:
            _set_station(lines, cell, t, 2)
    elif station == 2:  # SQUARING
        speed[cell, 2] = v_st
        if age >= lines.square and lines.act[cell, _REAR_BLADE] == lines.down:
            lines.act[cell, _REAR_BLADE] = lines.up
        if age >= lines.square + lines.blade_move:
            lines.act[cell, _CLAMP] = lines.clamp_closed
            _set_station(lines, cell, t, 3)
    elif station == 3:  # LOCKING
        speed[cell, 2] = v_st
        if age >= lines.clamp_move + lines.clamp_settle:
            # The source compares the float32 readback in float32.
            error = observation[cell, 14] - lines.clamp_on_box
            if wp.abs(error) <= lines.clamp_ok_window:
                lines.ready[cell] = 1
                lines.picks_done[cell] = 0
                _set_station(lines, cell, t, 4)
            else:
                _set_station(lines, cell, t, 6)
    elif station == 4:  # LOCKED
        if lines.picks_done[cell] >= 3 and lines.occ[cell, 3] < 0:
            lines.ready[cell] = 0
            lines.act[cell, _CLAMP] = lines.clamp_open
            lines.act[cell, _REAR_BLADE] = lines.down
            lines.act[cell, _FRONT_STOP] = lines.down
            _set_station(lines, cell, t, 5)
    elif station == 5:  # RELEASING
        if age >= lines.blade_move + wp.float64(0.10):
            speed[cell, 2] = v_tr
            speed[cell, 3] = wp.max(speed[cell, 3], v_tr)
        if lines.prev[cell, 2] != 0 and hits[cell, 2] == 0 and lines.occ[cell, 2] >= 0:
            _transfer(lines, cell, t, 2, 3)
            lines.front_up_at[cell] = t + lines.front_clear
            _set_station(lines, cell, t, 0)
    # The PreStop returns up once the fed box has cleared it.
    if not wp.isnan(lines.prestop_up_at[cell]) and t >= lines.prestop_up_at[cell]:
        lines.act[cell, _PRE_STOP] = lines.up
        lines.prestop_up_at[cell] = wp.float64(wp.nan)

    # Queue Z2.
    if lines.station[cell] != 1:
        if lines.occ[cell, 1] >= 0 and lines.hold[cell, 1] == 0:
            speed[cell, 1] = v_tr
            if hits[cell, 1] != 0:
                if wp.isnan(lines.seat_since[cell, 1]):
                    lines.seat_since[cell, 1] = t
                if t - lines.seat_since[cell, 1] >= lines.square:
                    lines.hold[cell, 1] = 1
                    speed[cell, 1] = wp.float64(0.0)
        if lines.occ[cell, 1] < 0:
            lines.hold[cell, 1] = 0
            lines.seat_since[cell, 1] = wp.float64(wp.nan)
    else:
        lines.hold[cell, 1] = 0
        lines.seat_since[cell, 1] = wp.float64(wp.nan)

    # Infeed Z1.
    if lines.occ[cell, 0] >= 0:
        if lines.occ[cell, 1] < 0:
            speed[cell, 0] = v_tr
            speed[cell, 1] = wp.max(speed[cell, 1], v_tr)
            if lines.prev[cell, 0] != 0 and hits[cell, 0] == 0:
                _transfer(lines, cell, t, 0, 1)
        elif hits[cell, 0] == 0:
            speed[cell, 0] = v_tr
    if t < lines.tail[cell, 0]:
        speed[cell, 0] = wp.max(speed[cell, 0], v_tr)
    for z in range(1, 3):
        if t < lines.tail[cell, z] and speed[cell, z] == wp.float64(0.0) and lines.occ[cell, z] < 0:
            speed[cell, z] = v_tr

    # CaseLine: end-stop hand-off into the sealer.
    if not wp.isnan(lines.end_up_at[cell]) and t >= lines.end_up_at[cell]:
        lines.act[cell, _END_STOP] = lines.up
        lines.end_up_at[cell] = wp.float64(wp.nan)
    release_started = bool(False)
    if (
        wp.isnan(lines.release8[cell])
        and lines.occ[cell, 7] >= 0
        and lines.hold[cell, 7] != 0
        and sealer_ready
        and wp.isnan(lines.end_up_at[cell])
    ):
        lines.release8[cell] = t
        release_started = True
        lines.act[cell, _END_STOP] = lines.down
    handed_off = int(-1)
    if not wp.isnan(lines.release8[cell]):
        if t - lines.release8[cell] >= lines.blade_move:
            speed[cell, 7] = v_tr
        if fell8 and lines.occ[cell, 7] >= 0:
            handed_off = lines.occ[cell, 7]
            lines.occ[cell, 7] = -1
            lines.hold[cell, 7] = 0
            lines.tail[cell, 7] = t + wp.float64(0.6)
            lines.seat_since[cell, 7] = wp.float64(wp.nan)
            lines.release8[cell] = wp.float64(wp.nan)
            lines.end_up_at[cell] = t + wp.float64(0.45)
    # CaseLineL02: the base controller raises the front stop only in EMPTY.
    if not wp.isnan(lines.front_up_at[cell]) and lines.station[cell] != 0 and t >= lines.front_up_at[cell]:
        lines.act[cell, _FRONT_STOP] = lines.up
        lines.front_up_at[cell] = wp.float64(wp.nan)

    if release_started:
        _set_sealer(lines, cell, t, 1)
    if handed_off >= 0:
        lines.sealer_box[cell] = handed_off

    # ---- Sealer.step --------------------------------------------------------------------------
    age = t - lines.sealer_t_state[cell]
    if not wp.isnan(lines.sealer_stop_up_at[cell]) and t >= lines.sealer_stop_up_at[cell]:
        lines.sealer_stop[cell] = lines.stop_up
        lines.sealer_stop_up_at[cell] = wp.float64(wp.nan)
    state = lines.sealer[cell]
    if state == 1:  # RECEIVE
        speed[cell, 8] = v_tr
        speed[cell, 9] = v_tr
        if hits[cell, _TS_SWAP] != 0:
            if wp.isnan(lines.sealer_seat_since[cell]):
                lines.sealer_seat_since[cell] = t
            speed[cell, 9] = v_st
            if t - lines.sealer_seat_since[cell] >= wp.float64(0.35):
                lines.sealer_seat_since[cell] = wp.float64(wp.nan)
                _set_sealer(lines, cell, t, 2)
        elif age > wp.float64(25.0):
            _set_sealer(lines, cell, t, 5)
    elif state == 2:  # PROCESS
        if age >= lines.process and lines.swap_request[cell] < 0:
            lines.swap_request[cell] = lines.sealer_box[cell]
    elif state == 3:  # SEALED
        if outfeed_ok[cell] != 0:
            lines.sealer_stop[cell] = lines.stop_down
            _set_sealer(lines, cell, t, 4)
    elif state == 4:  # RELEASE
        if age >= lines.blade_move:
            speed[cell, 9] = v_tr
        if lines.prev[cell, _TS_OUT] != 0 and hits[cell, _TS_OUT] == 0:
            lines.sealer_box[cell] = -1
            lines.sealer_stop_up_at[cell] = t + wp.float64(0.45)
            _set_sealer(lines, cell, t, 0)
        elif age > wp.float64(15.0):
            _set_sealer(lines, cell, t, 5)

    # Plant bindings: flaps close in PROCESS; the sealed-case swap is the tape bond.
    if lines.sealer[cell] == 2 and lines.sealer_box[cell] >= 0:
        closed[lines.sealer_box[cell]] = 1
    box = lines.swap_request[cell]
    if box >= 0 and sealed[box] != 0:
        lines.sealer_box[cell] = box
        lines.swap_request[cell] = -1
        _set_sealer(lines, cell, t, 3)

    # Case supply (CaseLine.can_spawn / on_spawn).
    spawn_enabled[cell] = 0
    if (
        lines.magazine[cell] > 0
        and lines.occ[cell, 0] < 0
        and hits[cell, _SPAWN] == 0
        and t - lines.last_spawn[cell] >= lines.spawn_interval
    ):
        box = lines.supply[cell, lines.supplied[cell]]
        lines.supplied[cell] += 1
        spawn_enabled[cell] = 1
        for slot in range(5):
            spawn_bodies[cell, slot] = case_bodies[box, slot]
            spawn_local[cell, slot] = case_local[box, slot]
        lines.occ[cell, 0] = box
        lines.hold[cell, 0] = 0
        lines.magazine[cell] -= 1
        lines.last_spawn[cell] = t

    for i in range(12):
        lines.prev[cell, i] = int(hits[cell, i] != 0)

    # MDR zone ramps toward the commanded surface speeds.
    for z in range(10):
        target = speed[cell, z]
        current = lines.zone_velocity[cell, z]
        ramp = lines.ramp_down
        if wp.abs(target) > wp.abs(current):
            ramp = lines.ramp_up
        delta = lines.v_tr / (ramp * wp.float64(60.0))
        lines.zone_velocity[cell, z] = current + wp.clamp(target - current, -delta, delta)

    line_ready[cell] = int(station_offers_case(lines.ready[cell], lines.station[cell], lines.picks_done[cell]))
    lines.status[cell, 0] = lines.station[cell]
    lines.status[cell, 1] = lines.sealer[cell]
    lines.status[cell, 2] = lines.magazine[cell]


@wp.kernel
def case_drive_targets(
    lines: CaseLines,
    cells: wp.array[int],
    kinds: wp.array[int],
    items: wp.array[int],
    closed_angle: wp.array[float],
    initial: wp.array[float],
    closed: wp.array[int],
    roller_radius: wp.float64,
    positions: wp.array[float],
    velocities: wp.array[float],
):
    """Map PLC outputs onto case-line roller, actuator and flap drive targets."""
    i = wp.tid()
    cell, item = cells[i], items[i]
    position = initial[i]
    velocity = float(0.0)
    if kinds[i] == 0:  # roller zone, or -1 for an undriven zone
        position = 0.0
        if item >= 0:
            velocity = float(lines.zone_velocity[cell, item] / roller_radius)
    elif kinds[i] == 1:  # actuator
        if item == 5:
            position = float(lines.sealer_stop[cell])
        else:
            position = float(lines.act[cell, item])
    elif closed[item] != 0:  # flap of a case that entered PROCESS
        position = closed_angle[i]
    positions[i] = position
    velocities[i] = velocity


def make_case_lines(design, controller, sealer, supplies, cells, device):
    """Allocate the device PLC state, seeded as the plant seeds the source lines.

    ``supplies`` lists each cell's reserve case indices in supply order; the
    initial case of each cell (``cells[i]``) is held on the queue zone Z2.
    """
    lines = CaseLines()
    n = len(supplies)
    nan = np.nan

    def full(shape, value, dtype):
        return wp.array(np.full(shape, value), dtype=dtype, device=device)

    lines.t = full(n, 0.0, wp.float64)
    lines.station = full(n, 0, int)
    lines.t_state = full(n, 0.0, wp.float64)
    lines.ready = full(n, 0, int)
    lines.picks_done = full(n, 0, int)
    lines.acknowledged = full(n, 0, int)
    occ = np.full((n, 8), -1)
    occ[:, 1] = cells
    hold = np.zeros((n, 8), dtype=np.int32)
    hold[:, 1] = 1
    lines.occ = wp.array(occ, dtype=int, device=device)
    lines.hold = wp.array(hold, dtype=int, device=device)
    lines.tail = full((n, 8), -1.0, wp.float64)
    lines.seat_since = full((n, 8), nan, wp.float64)
    lines.prev = full((n, 12), 0, int)
    act = [controller.UP, controller.DOWN, controller.UP, controller.CLAMP_OPEN, controller.UP]
    lines.act = wp.array(np.tile(act, (n, 1)), dtype=wp.float64, device=device)
    for name in ("prestop_up_at", "front_up_at", "release8", "end_up_at", "sealer_stop_up_at", "sealer_seat_since"):
        setattr(lines, name, full(n, nan, wp.float64))
    lines.magazine = wp.array([len(s) for s in supplies], dtype=int, device=device)
    lines.last_spawn = full(n, 2.0 - design.SPAWN["min_interval_s"], wp.float64)
    width = max(1, *(len(s) for s in supplies))
    lines.supply = wp.array([list(s) + [-1] * (width - len(s)) for s in supplies], dtype=int, device=device)
    lines.supplied = full(n, 0, int)
    lines.sealer = full(n, 0, int)
    lines.sealer_t_state = full(n, 0.0, wp.float64)
    lines.sealer_box = full(n, -1, int)
    lines.sealer_stop = full(n, sealer.STOP_UP, wp.float64)
    lines.swap_request = full(n, -1, int)
    lines.zone_velocity = full((n, len(ZONE_NAMES)), 0.0, wp.float64)
    lines.status = full((n, 3), 0, int)
    timing = design.TIMING
    lines.dt = 1.0 / 60.0
    lines.v_tr, lines.v_st = design.SPEED["transport"], design.SPEED["station"]
    lines.ramp_up, lines.ramp_down = design.RAMP_UP_S, design.RAMP_DOWN_S
    lines.square, lines.blade_move = timing["square_s"], timing["blade_move_s"]
    lines.clamp_move, lines.clamp_settle = timing["clamp_move_s"], timing["clamp_settle_s"]
    lines.front_clear, lines.prestop_clear = timing["front_clear_s"], timing["prestop_clear_s"]
    lines.endzone_settle = timing["endzone_settle_s"]
    lines.spawn_interval = design.SPAWN["min_interval_s"]
    lines.clamp_ok_window = design.CLAMP_OK_WINDOW
    lines.clamp_on_box = controller.CLAMP_ON_BOX
    lines.up, lines.down = controller.UP, controller.DOWN
    lines.clamp_open, lines.clamp_closed = controller.CLAMP_OPEN, controller.CLAMP_CLOSED
    lines.stop_up, lines.stop_down = sealer.STOP_UP, sealer.STOP_DOWN
    lines.process = sealer.process_s
    return lines
