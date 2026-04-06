#!/usr/bin/env python3
"""
A* Autonomous Navigator  --  3.0m × 1.2m mecanum robot
=======================================================
Combines the full motion-control stack (PI heading hold, per-wheel trim,
encoder distance, IMU turns) with an A* path planner so the car can
navigate around CSV-loaded obstacles to any clicked or typed goal.

Controls
--------
  Left-click          : set GOAL and run A* immediately (default mode)
  G                   : type goal as  x,y  then Enter
  S  / left-click(S)  : set-start mode  (place car start without planning)
  T                   : type start position  x,y
  L                   : load obstacle CSV (type path then Enter)
  R                   : replan A* from current position to last goal
  SPACE               : start / resume driving the planned path
  P                   : pause  (motors stop)
  C                   : clear path and stop
  1/2/3/4             : set facing +X / +Y / -X / -Y
  ESC / close         : safe stop + quit

How it works
------------
1. Load CSV ->> obstacles drawn on map, occupancy grid rebuilt.
2. Left-click anywhere on the map ->> that point is the goal.
   A* runs from current car position, path smoothed with line-of-sight
   string-pulling, then loaded as the waypoint queue.
3. Press SPACE ->> car executes each waypoint leg using:
     - IMU-based PI heading hold (minimises drift on straight legs)
     - Per-wheel PWM trim (compensates individual motor variation)
     - All-4-encoder distance averaging (mecanum slip robustness)
     - Post-turn IMU anchor (resets yaw reference after each turn)
4. Press R at any time to replan from wherever the car is now.
"""

import time, math, os, csv, heapq, serial, pygame
from collections import deque
from smbus2 import SMBus

# =====================================================================
# Hardware
# =====================================================================
PORT           = "/dev/ttyUSB0"
BAUD           = 115200
SER_TIMEOUT    = 0.05
UPLOAD_ENABLE  = "$upload:1,1,1#"
UPLOAD_DISABLE = "$upload:0,0,0#"
MOTOR_TYPE_CMD = "$mtype:3#"

# =====================================================================
# Car dimensions
# =====================================================================
CAR_LENGTH_M = 0.24    # front-to-back (m)
CAR_WIDTH_M  = 0.20    # side-to-side  (m)
# Diagonal half-length used to inflate obstacles in the planner so the
# car body never collides even when approaching at an angle.
CAR_RADIUS_M = 0.5 * math.hypot(CAR_LENGTH_M, CAR_WIDTH_M)   # ≈ 0.156 m

# =====================================================================
# CSV bounding-box scale
# =====================================================================
# Sensor/LiDAR CSV bounding boxes are often larger than the physical
# obstacle.  BOX_SCALE multiplies every box's width and length before
# drawing and before building the occupancy grid.
# 1.0 = use CSV values as-is.  0.5 = half the CSV size.
# Adjust with [ / ] keys at runtime.
BOX_SCALE      = 1.0    # default scale applied to all CSV box dimensions
BOX_SCALE_STEP = 0.05   # how much each [ / ] press changes the scale
BOX_SCALE_MIN  = 0.10
BOX_SCALE_MAX  = 3.00

# =====================================================================
# Obstacle CSV
# =====================================================================
# Default CSV loaded automatically on startup.
# Change this path to point at your obstacle file, or press L at
# runtime to load a different file without restarting.
BBOX_CSV_PATH = "boundingboxes.csv"

# =====================================================================
# Calibrations
# =====================================================================
WHEEL_DIAMETER_M     = 0.08
COUNTS_PER_WHEEL_REV = 560

# =====================================================================
# Motor mapping
# =====================================================================
FL, RL, FR, RR = 1, 2, 3, 4

# =====================================================================
# Motion tuning
# =====================================================================
START_PWM  = 1200
CRUISE_PWM = 1200
KICK_TIME  = 0.20

FL_TRIM = 1.00
RL_TRIM = 1.00
FR_TRIM = 0.9145
RR_TRIM = 0.9145

# PI heading hold
HEADING_KP            = 3.0
HEADING_KI            = 0.8
HEADING_KI_MAX        = 150.0
HEADING_CORR_DEADBAND = 0.5     # degrees -- ignore IMU noise below this
MAX_CORR              = 400
INVERT_YAW            = False

# Distance calibration
DIST_SCALE   = 1.02#1.1
STOP_EARLY_M = 0.01

# Turn controller
TURN_STOP_EPS_DEG = 0.7
TURN_IMU_KP       = 18.0#12
TURN_MAX_PWM      = 1200
TURN_MIN_PWM      = 1000#1100
TURN_KICK         = 1200#1400
TURN_KICK_TIME    = 0.07#0.1
TURN_TIMEOUT_S    = 5.0
ANGLE_EPS_DEG     = 1.0
TURN_SETTLE_S = 0.15    # pause after each turn to let IMU settle (seconds)
# Drive timeouts
ENCODER_TIMEOUT_S  = 1.0
MIN_DRIVE_SPEED_MS = 0.10
DRIVE_TIMEOUT_PAD_S = 4.0
ENC_DRAIN_LINES    = 8

# Waypoint arrival radius -- pop waypoint when within this distance
WP_ARRIVE_M = 0.04

# =====================================================================
# IMU
# =====================================================================
I2C_BUS          = 1
IMU_ADDR         = 0x68
PWR_MGMT_1       = 0x6B
WHO_AM_I         = 0x75
GYRO_ZOUT_H      = 0x47
GYRO_LSB_PER_DPS = 131.0
BIAS_CAL_TIME    = 2.0

# =====================================================================
# Map / UI
# =====================================================================
WIDTH, HEIGHT = 900, 650
MARGIN_PX     = 60
WORLD_W_M     = 3.0
WORLD_H_M     = 1.2
HALF_W        = WORLD_W_M / 2.0   # 1.5 m
HALF_H        = WORLD_H_M / 2.0   # 0.6 m

_PPM = min((WIDTH  - 2 * MARGIN_PX) / WORLD_W_M,
           (HEIGHT - 2 * MARGIN_PX) / WORLD_H_M)
_CX  = WIDTH  / 2.0
_CY  = HEIGHT / 2.0

# =====================================================================
# Circle obstacles (placed interactively)
# =====================================================================
# All interactively placed obstacles are standardised circles of this radius.
CIRCLE_R_M = 0.08   # 8 cm radius
GRID_RES    = 0.05    # metres per grid cell -- finer = more accurate but slower
ALLOW_DIAG  = True    # allow 45-degree diagonal moves
NO_CORNER_CUT = True  # prevent squeezing through diagonal gaps between obstacles
LOS_STEP    = 0.01    # line-of-sight sampling step (m) for path smoothing

# =====================================================================
# Colours
# =====================================================================
BG        = (255, 255, 255)
GRID_C    = (230, 230, 230)
BOLD_GRID = (210, 210, 210)
AXIS      = (160, 160, 160)
BLACK     = (  0,   0,   0)
BLUE      = ( 40,  90, 255)
RED       = (220,  60,  60)
GREEN     = ( 40, 160,  70)
PURPLE    = (150,  80, 190)
DARK      = ( 60,  60,  60)
WHITE     = (255, 255, 255)
ORANGE    = (255, 140,   0)
ORANGE_L  = (255, 180,  80)
CYAN      = ( 80, 200, 220)
CIRCLE_C  = (200,  40, 200)   # magenta -- circle obstacles placed interactively
CIRCLE_CL = (220, 140, 220)   # lighter magenta for label / inflated ring
BOX_DRAG  = ( 80, 200,  80)   # bright green -- box being dragged
PATH_COL  = (  0, 160,  70)
EXP_COL   = (180, 210, 255)   # explored-cell tint
GOAL_COL  = (220,  40,  40)
CAR_BODY  = ( 40,  90, 255)
CAR_FRONT = (220,  60,  60)
CAR_CTRE  = (255, 255, 255)

PI = math.pi

# =====================================================================
# Utility
# =====================================================================
def clamp(v, lo, hi):          return max(lo, min(hi, v))
def clamp_world(x, y):         return (clamp(x, -HALF_W, HALF_W),
                                        clamp(y, -HALF_H, HALF_H))
def wrap_deg(a):               return (a + 180.0) % 360.0 - 180.0
def counts_to_m(c):            return (c / COUNTS_PER_WHEEL_REV) * (PI * WHEEL_DIAMETER_M)
def to_pygame(xy):
    x, y = xy
    return int(_CX + x * _PPM), int(_CY - y * _PPM)
def from_pygame(pxy):
    px, py = pxy
    return clamp_world((px - _CX) / _PPM, -(py - _CY) / _PPM)

# =====================================================================
# Font cache
# =====================================================================
_font_cache = {}
def _font(sz):
    if sz not in _font_cache:
        _font_cache[sz] = pygame.font.SysFont("consolas", sz)
    return _font_cache[sz]
def draw_text(screen, txt, x, y, col=BLACK, sz=18):
    screen.blit(_font(sz).render(txt, True, col), (x, y))

# =====================================================================
# Serial helpers
# =====================================================================
def send(ser, cmd):
    if not (cmd.startswith("$") and cmd.endswith("#")):
        raise ValueError(f"Bad cmd: {cmd!r}")
    ser.write(cmd.encode("ascii")); ser.flush()

def pwm_cmd(ser, m1, m2, m3, m4):
    send(ser, f"$pwm:{int(m1)},{int(m2)},{int(m3)},{int(m4)}#")

def stop_motors(ser):
    if ser.is_open:
        pwm_cmd(ser, 0, 0, 0, 0)
        time.sleep(0.05)

def parse_mall(line):
    line = line.strip()
    if not line.startswith("$MAll:"): return None
    body = line[6:].rstrip("#")
    parts = body.split(",")
    if len(parts) < 4: return None
    try:    return tuple(int(p) for p in parts[:4])
    except: return None

def wait_for_mall(ser, timeout=1.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:    line = ser.readline().decode("ascii", "ignore")
        except: break
        v = parse_mall(line)
        if v is not None: return v
    return None

def drain_mall(ser, current, max_lines=ENC_DRAIN_LINES):
    latest = current
    for _ in range(max_lines):
        if ser.in_waiting == 0: break
        try:    line = ser.readline().decode("ascii", "ignore")
        except: break
        v = parse_mall(line)
        if v is not None: latest = v
    return latest

def enc_delta(cur, start):
    return tuple(cur[i] - start[i] for i in range(4))

def all_wheel_avg_m(delta):
    return (sum(counts_to_m(abs(delta[i])) for i in range(4)) / 4.0) * DIST_SCALE

def drive_fwd(ser, base_pwm, corr):
    pwm_cmd(ser,
            int(base_pwm * FL_TRIM - corr),
            int(base_pwm * RL_TRIM - corr),
            int(base_pwm * FR_TRIM + corr),
            int(base_pwm * RR_TRIM + corr))

# =====================================================================
# IMU
# =====================================================================
def read_i16(bus, addr, reg):
    data = bus.read_i2c_block_data(addr, reg, 2)
    v = (data[0] << 8) | data[1]
    return v - 65536 if v & 0x8000 else v

class IMUYaw:
    def __init__(self, bus):
        self.bus = bus; self.yaw_deg = 0.0
        self.bias_dps = 0.0; self._last_t = time.time()

    def reset_pose(self, yaw=0.0):
        self.yaw_deg = yaw; self._last_t = time.time()

    def init(self):
        who = self.bus.read_byte_data(IMU_ADDR, WHO_AM_I)
        print(f"IMU WHO_AM_I=0x{who:02X}")
        self.bus.write_byte_data(IMU_ADDR, PWR_MGMT_1, 0x00)
        time.sleep(0.05); self._last_t = time.time()

    def calibrate_bias(self, seconds=2.0):
        print(f"Calibrating gyro bias {seconds:.1f}s -- keep robot still.")
        t0 = time.time(); acc = 0.0; n = 0
        while time.time() - t0 < seconds:
            acc += read_i16(self.bus, IMU_ADDR, GYRO_ZOUT_H) / GYRO_LSB_PER_DPS
            n += 1; time.sleep(0.01)
        self.bias_dps = acc / n if n else 0.0
        print(f"Gyro Z bias ≈ {self.bias_dps:.4f} dps")
        self.reset_pose(0.0)

    def update(self):
        now = time.time(); dt = now - self._last_t; self._last_t = now
        try:    gz_raw = read_i16(self.bus, IMU_ADDR, GYRO_ZOUT_H)
        except: return self.yaw_deg
        self.yaw_deg = wrap_deg(
            self.yaw_deg + (gz_raw / GYRO_LSB_PER_DPS - self.bias_dps) * dt)
        return self.yaw_deg

# =====================================================================
# Motion primitives  (unchanged from main script)
# =====================================================================
def do_turn_to_heading_imu(ser, imu, dtheta_deg):
    """In-place turn by dtheta_deg using IMU feedback."""
    if abs(dtheta_deg) < ANGLE_EPS_DEG:
        return

    start_yaw  = imu.update()
    target_yaw = wrap_deg(start_yaw + dtheta_deg)

    # Kick to break static friction
    if dtheta_deg > 0:
        pwm_cmd(ser, -TURN_KICK, -TURN_KICK, +TURN_KICK, +TURN_KICK)
    else:
        pwm_cmd(ser, +TURN_KICK, +TURN_KICK, -TURN_KICK, -TURN_KICK)
    time.sleep(TURN_KICK_TIME)

    t0 = time.time()
    while True:
        if time.time() - t0 > TURN_TIMEOUT_S:
            print("Turn timeout."); break
        yaw = imu.update()
        err = wrap_deg(target_yaw - yaw)
        if abs(err) <= TURN_STOP_EPS_DEG: break
        mag = clamp(abs(TURN_IMU_KP * err), TURN_MIN_PWM, TURN_MAX_PWM)
        if err > 0: pwm_cmd(ser, -mag, -mag, +mag, +mag)
        else:       pwm_cmd(ser, +mag, +mag, -mag, -mag)

    stop_motors(ser)
    time.sleep(0.03)


def do_drive_distance_with_imu(ser, imu, dist_m, target_heading_deg):
    """
    Drive forward dist_m metres holding target_heading_deg.
    Uses PI heading correction + all-4-encoder averaging.
    """
    if dist_m <= 1e-4:
        return

    target_m    = max(0.0, dist_m - STOP_EARLY_M)
    fwd_timeout = dist_m / MIN_DRIVE_SPEED_MS + DRIVE_TIMEOUT_PAD_S

    ser.reset_input_buffer()
    drive_fwd(ser, START_PWM, 0.0)
    time.sleep(KICK_TIME)

    ser.reset_input_buffer()
    start_enc = wait_for_mall(ser, ENCODER_TIMEOUT_S)
    if start_enc is None:
        stop_motors(ser)
        raise RuntimeError("No $MAll stream after kick.")

    latest = start_enc
    t0 = time.time(); last_t = t0; heading_i = 0.0

    while True:
        now = time.time(); dt = now - last_t; last_t = now

        if now - t0 > fwd_timeout:
            delta = enc_delta(latest, start_enc)
            print(f"Drive timeout -- covered "
                  f"{all_wheel_avg_m(delta):.2f}/{dist_m:.2f} m")
            break

        yaw = imu.update()
        err = wrap_deg(target_heading_deg - yaw)
        if abs(err) < HEADING_CORR_DEADBAND:
            err = 0.0

        heading_i = clamp(heading_i + err * dt, -HEADING_KI_MAX, HEADING_KI_MAX)
        corr = clamp(HEADING_KP * err + HEADING_KI * heading_i, -MAX_CORR, MAX_CORR)
        if INVERT_YAW: corr = -corr

        drive_fwd(ser, CRUISE_PWM, corr)

        latest = drain_mall(ser, latest)
        if all_wheel_avg_m(enc_delta(latest, start_enc)) >= target_m:
            break

    stop_motors(ser)
    time.sleep(0.05)

# =====================================================================
# Obstacle CSV loader
# =====================================================================
def load_boxes(path: str) -> list:
    """
    Load rotated bounding boxes from CSV.
    Columns: cx, cy, cz(ignored), width, length, height(ignored),
             rot_x(ignored), rot_y(ignored), rot_z_deg
    Returns list of dicts with keys: cx, cy, width, length, rot_z_deg, label
    """
    boxes = []
    if not os.path.isfile(path):
        print(f"CSV not found: {path!r}"); return boxes
    with open(path, newline="") as f:
        for row_num, row in enumerate(csv.reader(f), 1):
            if len(row) < 9: continue
            try:
                cx = float(row[0]); cy = float(row[1])
                w  = float(row[3]); l  = float(row[4])
                rz = float(row[8])
            except ValueError:
                continue
            boxes.append({"cx": cx, "cy": cy, "width": w,
                           "length": l, "rot_z_deg": rz,
                           "label": str(len(boxes) + 1)})
    print(f"Loaded {len(boxes)} box(es) from {path!r}")
    return boxes

def box_corners(b, scale=1.0):
    """4 world-space corners of a rotated rectangle, optionally scaled."""
    cx, cy  = b["cx"], b["cy"]
    w, l    = b["width"] * scale, b["length"] * scale
    hw, hl  = w / 2, l / 2
    rad     = math.radians(b["rot_z_deg"])
    c, s    = math.cos(rad), math.sin(rad)
    return [(cx + lx*c - ly*s, cy + lx*s + ly*c)
            for lx, ly in [( hw, hl), (-hw, hl), (-hw,-hl), ( hw,-hl)]]

# =====================================================================
# A* planner
# =====================================================================
def _grid_dims():
    nx = int(round(WORLD_W_M / GRID_RES)) + 1
    ny = int(round(WORLD_H_M / GRID_RES)) + 1
    return nx, ny

def _w2c(x, y):
    """World coords ->> grid cell."""
    return (int(round((x + HALF_W) / GRID_RES)),
            int(round((y + HALF_H) / GRID_RES)))

def _c2w(ix, iy):
    """Grid cell ->> world coords."""
    return (-HALF_W + ix * GRID_RES, -HALF_H + iy * GRID_RES)

def _point_in_obb(px, py, cx, cy, w, l, ang):
    dx, dy = px - cx, py - cy
    c, s = math.cos(-ang), math.sin(-ang)
    return abs(c*dx - s*dy) <= w/2 and abs(s*dx + c*dy) <= l/2

def build_occupancy(boxes: list, circles: list = None, box_scale: float = 1.0) -> list:
    """
    Build a 2-D occupancy grid inflated by CAR_RADIUS_M.

    boxes      -- rotated-rectangle obstacles loaded from CSV
    circles    -- list of (cx, cy) tuples for interactively placed circle obstacles
    box_scale  -- multiplier applied to each box's width/length before inflation
    """
    nx, ny = _grid_dims()
    occ = [[False] * nx for _ in range(ny)]

    # ── Rotated-box obstacles (from CSV) ─────────────────────────────
    for b in boxes:
        cx, cy = b["cx"], b["cy"]
        w = b["width"]  * box_scale + 2 * CAR_RADIUS_M
        l = b["length"] * box_scale + 2 * CAR_RADIUS_M
        ang = math.radians(b["rot_z_deg"])
        r = 0.5 * math.hypot(w, l)
        ix0, iy0 = _w2c(cx - r, cy - r)
        ix1, iy1 = _w2c(cx + r, cy + r)
        ix0, ix1 = sorted((ix0, ix1)); iy0, iy1 = sorted((iy0, iy1))
        ix0 = max(0, min(ix0, nx-1)); ix1 = max(0, min(ix1, nx-1))
        iy0 = max(0, min(iy0, ny-1)); iy1 = max(0, min(iy1, ny-1))
        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                px, py = _c2w(ix, iy)
                if _point_in_obb(px, py, cx, cy, w, l, ang):
                    occ[iy][ix] = True

    # ── Circle obstacles (placed interactively) ───────────────────────
    if circles:
        inflate_r = CIRCLE_R_M + CAR_RADIUS_M   # physical + safety margin
        for (cx, cy) in circles:
            ix0, iy0 = _w2c(cx - inflate_r, cy - inflate_r)
            ix1, iy1 = _w2c(cx + inflate_r, cy + inflate_r)
            ix0, ix1 = sorted((ix0, ix1)); iy0, iy1 = sorted((iy0, iy1))
            ix0 = max(0, min(ix0, nx-1)); ix1 = max(0, min(ix1, nx-1))
            iy0 = max(0, min(iy0, ny-1)); iy1 = max(0, min(iy1, ny-1))
            for iy in range(iy0, iy1 + 1):
                for ix in range(ix0, ix1 + 1):
                    px, py = _c2w(ix, iy)
                    if math.hypot(px - cx, py - cy) <= inflate_r:
                        occ[iy][ix] = True

    return occ

def _nearest_free(cell, occ):
    """BFS from cell to find nearest unoccupied grid cell."""
    nx = len(occ[0]); ny = len(occ)
    def free(c):
        x, y = c
        return 0 <= x < nx and 0 <= y < ny and not occ[y][x]
    if free(cell): return cell
    q = deque([cell]); vis = {cell}
    while q:
        x, y = q.popleft()
        if free((x, y)): return (x, y)
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            n = (x+dx, y+dy)
            if n not in vis: vis.add(n); q.append(n)
    return None

def _neighbours(n, occ):
    x, y = n
    nx = len(occ[0]); ny = len(occ)
    steps = [(1,0),(-1,0),(0,1),(0,-1)]
    if ALLOW_DIAG: steps += [(1,1),(1,-1),(-1,1),(-1,-1)]
    for dx, dy in steps:
        x2, y2 = x+dx, y+dy
        if not (0 <= x2 < nx and 0 <= y2 < ny): continue
        if occ[y2][x2]: continue
        if ALLOW_DIAG and NO_CORNER_CUT and dx and dy:
            if occ[y][x+dx] or occ[y+dy][x]: continue
        yield (x2, y2)

def _astar_cells(s, g, occ):
    """Raw A* returning (path_cells, explored_cells)."""
    pq = [(0.0, s)]; heapq.heapify(pq)
    came = {}; cost = {s: 0.0}; closed = set(); explored = []
    while pq:
        _, cur = heapq.heappop(pq)
        if cur in closed: continue
        closed.add(cur); explored.append(cur)
        if cur == g:
            path = [cur]
            while cur in came: cur = came[cur]; path.append(cur)
            path.reverse()
            return path, explored
        for nb in _neighbours(cur, occ):
            ng = cost[cur] + math.hypot(nb[0]-cur[0], nb[1]-cur[1])
            if nb not in cost or ng < cost[nb]:
                cost[nb] = ng; came[nb] = cur
                heapq.heappush(pq, (ng + math.hypot(nb[0]-g[0], nb[1]-g[1]), nb))
    return None, explored

def _point_free(x, y, occ):
    ix, iy = _w2c(x, y)
    nx = len(occ[0]); ny = len(occ)
    if not (0 <= ix < nx and 0 <= iy < ny): return False
    return not occ[iy][ix]

def _line_of_sight(a, b, occ):
    """Check a->>b straight line is collision-free."""
    ax, ay = a; bx, by = b
    d = math.hypot(bx-ax, by-ay)
    if d < 1e-9: return True
    n = max(2, int(d / LOS_STEP))
    for i in range(n + 1):
        t = i / n
        if not _point_free(ax + (bx-ax)*t, ay + (by-ay)*t, occ):
            return False
    return True

def _simplify(path):
    """Remove collinear points (keep direction-change nodes only)."""
    if not path or len(path) < 3: return path
    def _dir(a, b):
        dx = b[0]-a[0]; dy = b[1]-a[1]
        sx = 0 if abs(dx)<1e-9 else (1 if dx>0 else -1)
        sy = 0 if abs(dy)<1e-9 else (1 if dy>0 else -1)
        return (sx, sy)
    out = [path[0]]; pd = _dir(path[0], path[1])
    for i in range(1, len(path)-1):
        d = _dir(path[i], path[i+1])
        if d != pd: out.append(path[i]); pd = d
    out.append(path[-1])
    return out

def _smooth_los(path, occ):
    """String-pulling: skip intermediate nodes where line-of-sight is clear."""
    if not path or len(path) < 3: return path
    out = [path[0]]; i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _line_of_sight(path[i], path[j], occ):
            j -= 1
        out.append(path[j]); i = j
    return out

def plan_path(start_xy, goal_xy, occ):
    """
    Run A* from start_xy to goal_xy.
    Returns (waypoints_world, explored_world) or (None, explored_world).
    waypoints_world is the smoothed, reduced path ready for the car to follow.
    """
    sc = _nearest_free(_w2c(*start_xy), occ)
    gc = _nearest_free(_w2c(*goal_xy),  occ)
    if sc is None or gc is None:
        print("Start or goal has no reachable free cell.")
        return None, []

    cells, exp_cells = _astar_cells(sc, gc, occ)
    exp_world = [_c2w(*c) for c in exp_cells]

    if cells is None:
        print("A* found no path.")
        return None, exp_world

    path = [_c2w(*c) for c in cells]
    path = _simplify(path)
    path = _smooth_los(path, occ)
    # Always ensure the exact goal coordinate is the last waypoint
    path[-1] = goal_xy
    return path, exp_world

# =====================================================================
# Drawing
# =====================================================================
def draw_grid(screen):
    screen.fill(BG)
    left,  top    = to_pygame((-HALF_W,  HALF_H))
    right, bottom = to_pygame(( HALF_W, -HALF_H))
    pygame.draw.rect(screen, AXIS,
                     pygame.Rect(left, top, right-left, bottom-top), 2)

    def _lines(step, col, lw):
        for i in range(round(WORLD_W_M / step) + 1):
            xv = -HALF_W + i * step
            pygame.draw.line(screen, col,
                             to_pygame((xv,  HALF_H)),
                             to_pygame((xv, -HALF_H)), lw)
        for j in range(round(WORLD_H_M / step) + 1):
            yv = -HALF_H + j * step
            pygame.draw.line(screen, col,
                             to_pygame((-HALF_W, yv)),
                             to_pygame(( HALF_W, yv)), lw)

    _lines(0.1, GRID_C,    1)
    _lines(0.5, BOLD_GRID, 2)
    pygame.draw.line(screen, AXIS, to_pygame((0,  HALF_H)), to_pygame((0, -HALF_H)), 2)
    pygame.draw.line(screen, AXIS, to_pygame((-HALF_W, 0)), to_pygame((HALF_W, 0)), 2)

    for xv in (-0.75, 0.0, 0.75):
        px, py = to_pygame((xv, 0.0))
        draw_text(screen, "0" if xv==0 else f"{xv:+.2f}", px-16, py+8, DARK, 16)
    for yv in (-HALF_H, 0.0, HALF_H):
        px, py = to_pygame((0.0, yv))
        draw_text(screen, "0" if yv==0 else f"{yv:+.2f}", px+8, py-10, DARK, 16)
    rx, ry = to_pygame((HALF_W,  0.0)); draw_text(screen, "x (m)", rx-45, ry+28, DARK, 18)
    tx, ty = to_pygame((0.0,  HALF_H)); draw_text(screen, "y (m)", tx+10, ty+8,  DARK, 18)


def draw_boxes(screen, boxes, box_scale=1.0, drag_idx=None, box_move_mode=False):
    if not boxes: return
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for i, b in enumerate(boxes, 1):
        is_drag = (drag_idx is not None and drag_idx == i - 1)
        fill_col  = (80, 200, 80, 100)  if is_drag else (255, 140, 0, 80)
        edge_col  = BOX_DRAG            if is_drag else ORANGE
        pts = [to_pygame(p) for p in box_corners(b, box_scale)]

        # Check if box centre is visible in the map window
        cx_px, cy_px = to_pygame((b["cx"], b["cy"]))
        map_left, map_top    = to_pygame((-HALF_W,  HALF_H))
        map_right, map_bot   = to_pygame(( HALF_W, -HALF_H))
        on_screen = (map_left <= cx_px <= map_right and map_top <= cy_px <= map_bot)

        if not on_screen:
            # Box is off-screen -- draw a warning arrow at the nearest map edge
            edge_x = clamp(cx_px, map_left + 5, map_right  - 5)
            edge_y = clamp(cy_px, map_top  + 5, map_bot    - 5)
            pygame.draw.circle(screen, RED, (edge_x, edge_y), 10, 2)
            draw_text(screen, f"Box {i} off-screen",
                      edge_x + 12, edge_y - 8, RED, 13)
            continue   # skip drawing the actual polygon off-screen

        pygame.draw.polygon(overlay, fill_col, pts)
        pygame.draw.polygon(screen,  edge_col, pts, 3 if not is_drag else 2)
        # Centre handle -- larger and brighter in box-move mode so it's easy to grab
        handle_r = 7 if box_move_mode else 4
        handle_c = BOX_DRAG if is_drag else ((255, 180, 30) if box_move_mode else (220, 80, 0))
        pygame.draw.circle(screen, handle_c, (cx_px, cy_px), handle_r)
        if box_move_mode:
            pygame.draw.circle(screen, handle_c, (cx_px, cy_px), handle_r, 2)
        draw_text(screen, b["label"], cx_px + 8, cy_px - 16, ORANGE_L if not is_drag else BOX_DRAG, 13)
        # Show scaled size alongside label when scale != 1
        if abs(box_scale - 1.0) > 0.01:
            sw = b["width"]  * box_scale
            sl = b["length"] * box_scale
            draw_text(screen, f"{sw*100:.0f}x{sl*100:.0f}cm",
                      cx_px + 8, cy_px - 2, ORANGE_L, 11)
    screen.blit(overlay, (0, 0))


def draw_circle_obstacles(screen, circles: list, circle_mode: bool) -> None:
    """
    Draw interactively placed circle obstacles.
    - Solid magenta filled circle  = physical obstacle (CIRCLE_R_M radius)
    - Dashed magenta ring          = inflated safety zone (+ CAR_RADIUS_M)
    - Number label on each circle
    - Crosshair cursor on the map when circle_mode is active
    """
    inflate_r = CIRCLE_R_M + CAR_RADIUS_M
    for i, (cx, cy) in enumerate(circles, 1):
        px, py = to_pygame((cx, cy))
        r_px       = int(CIRCLE_R_M  * _PPM)
        inflate_px = int(inflate_r   * _PPM)

        # Filled physical body
        surf = pygame.Surface((r_px*2+2, r_px*2+2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*CIRCLE_C, 100), (r_px+1, r_px+1), r_px)
        screen.blit(surf, (px - r_px - 1, py - r_px - 1))

        # Outline
        pygame.draw.circle(screen, CIRCLE_C, (px, py), r_px, 2)

        # Inflated safety ring (dashed approximation: dotted circle)
        for angle_deg in range(0, 360, 12):
            a = math.radians(angle_deg)
            dx, dy = int(inflate_px * math.cos(a)), int(inflate_px * math.sin(a))
            pygame.draw.circle(screen, CIRCLE_CL, (px + dx, py + dy), 2)

        # Centre dot
        pygame.draw.circle(screen, CIRCLE_C, (px, py), 3)

        # Label
        draw_text(screen, str(i), px + r_px + 3, py - 10, CIRCLE_CL, 13)

    # Crosshair cursor when placing mode is active
    if circle_mode:
        mx, my = pygame.mouse.get_pos()
        r = int(CIRCLE_R_M * _PPM)
        pygame.draw.circle(screen, CIRCLE_C, (mx, my), r, 1)
        pygame.draw.line(screen, CIRCLE_C, (mx - r - 6, my), (mx + r + 6, my), 1)
        pygame.draw.line(screen, CIRCLE_C, (mx, my - r - 6), (mx, my + r + 6), 1)


def draw_goal_queue(screen, goal_queue: list) -> None:
    """
    Draw all queued goals as numbered markers.
    Goal #1 (active) = large red X with ring.
    Goals #2+ (pending) = smaller orange circles with sequence numbers.
    A thin dashed line connects them in order so the full planned route is visible.
    """
    if not goal_queue:
        return

    # Connecting line between queued goals (start ->> g1 ->> g2 ->> ...)
    if len(goal_queue) >= 2:
        pts = [to_pygame(g) for g in goal_queue]
        for i in range(len(pts) - 1):
            # Draw dashed segment
            x1, y1 = pts[i]; x2, y2 = pts[i + 1]
            dx, dy = x2 - x1, y2 - y1
            dist = math.hypot(dx, dy)
            if dist < 1: continue
            steps = max(2, int(dist / 12))
            for s in range(steps):
                if s % 2 == 0:
                    tx1 = x1 + dx * (s / steps)
                    ty1 = y1 + dy * (s / steps)
                    tx2 = x1 + dx * ((s + 1) / steps)
                    ty2 = y1 + dy * ((s + 1) / steps)
                    pygame.draw.line(screen, (200, 100, 50),
                                     (int(tx1), int(ty1)), (int(tx2), int(ty2)), 1)

    for i, (gx, gy) in enumerate(goal_queue):
        px, py = to_pygame((gx, gy))
        if i == 0:
            # Active goal -- red X with a ring
            pygame.draw.circle(screen, GOAL_COL, (px, py), 11, 2)
            pygame.draw.line(screen, GOAL_COL, (px-8, py-8), (px+8, py+8), 3)
            pygame.draw.line(screen, GOAL_COL, (px-8, py+8), (px+8, py-8), 3)
            draw_text(screen, "G1", px + 13, py - 10, GOAL_COL, 14)
        else:
            # Pending goals -- orange circle with number
            col = (220, 130, 30)
            pygame.draw.circle(screen, col, (px, py), 7, 2)
            draw_text(screen, f"G{i+1}", px + 9, py - 9, col, 13)


def draw_explored(screen, exp_world, surface_cache):
    """Draw A* explored cells as a faint blue dot layer (cached surface)."""
    if surface_cache[0] is None and exp_world:
        surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for x, y in exp_world:
            px, py = to_pygame((x, y))
            if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                surf.set_at((px, py), (*EXP_COL, 160))
        surface_cache[0] = surf
    if surface_cache[0]:
        screen.blit(surface_cache[0], (0, 0))


def draw_path(screen, path_world, current_wp_idx):
    """Draw the A* path as a green line; highlight the next waypoint."""
    if not path_world or len(path_world) < 2: return
    pts = [to_pygame(p) for p in path_world]
    pygame.draw.lines(screen, PATH_COL, False, pts, 2)
    for i, p in enumerate(pts):
        col = GREEN if i == current_wp_idx else (180, 220, 180)
        pygame.draw.circle(screen, col, p, 5 if i == current_wp_idx else 3)
    # Goal marker (red X)
    gx, gy = pts[-1]
    pygame.draw.line(screen, GOAL_COL, (gx-8, gy-8), (gx+8, gy+8), 3)
    pygame.draw.line(screen, GOAL_COL, (gx-8, gy+8), (gx+8, gy-8), 3)


def draw_robot(screen, x, y, yaw_deg):
    hl = CAR_LENGTH_M / 2.0; hw = CAR_WIDTH_M / 2.0
    rad = math.radians(yaw_deg)
    c, s = math.cos(rad), math.sin(rad)
    local = [( hl, hw), ( hl,-hw), (-hl,-hw), (-hl, hw)]
    pts = [to_pygame((x + lx*c - ly*s, y + lx*s + ly*c)) for lx, ly in local]
    pygame.draw.polygon(screen, CAR_BODY,  pts)
    pygame.draw.polygon(screen, CAR_FRONT, pts, 2)
    pygame.draw.line(screen, CAR_FRONT, pts[0], pts[1], 3)
    cx_px, cy_px = to_pygame((x, y))
    pygame.draw.line(screen, CAR_CTRE, (cx_px-5, cy_px), (cx_px+5, cy_px), 2)
    pygame.draw.line(screen, CAR_CTRE, (cx_px, cy_px-5), (cx_px, cy_px+5), 2)


def draw_input_overlay(screen, mode, buf, active_csv_path="", boxes=None, box_scale=1.0):
    overlay = pygame.Surface((WIDTH, 110), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160)); screen.blit(overlay, (0, 0))

    prompts = {
        "GOAL":       "Add GOAL -- type: x,y   then Enter  (ESC cancels)",
        "START":      "Set START -- type: x,y  then Enter  (ESC cancels)",
        "CSV":        f"CSV path (current: {active_csv_path or 'none'})  then Enter  (ESC cancels)",
        "CIRCLE":     "Place circle -- type: x,y   then Enter  (ESC cancels)",
        "BOX_CENTRE": "Move box centre -- type: box_num  x,y   e.g.  1  0.5,0.1   (ESC cancels)",
        "BOX_SCALE":  f"Set box scale (current: {box_scale:.3f}) -- type a number e.g. 0.25  (ESC cancels)",
    }
    draw_text(screen, prompts.get(mode, ""), 10, 8, WHITE, 17)

    # For BOX_CENTRE show a quick reference of current box positions
    if mode == "BOX_CENTRE" and boxes:
        summary = "  ".join(
            f"[{i+1}] ({b['cx']:+.2f},{b['cy']:+.2f})"
            for i, b in enumerate(boxes)
        )
        draw_text(screen, summary[:110], 10, 32, (200, 200, 200), 14)

    draw_text(screen, f"> {buf}", 10, 58, WHITE, 28)

# =====================================================================
# Heading name helper
# =====================================================================
def heading_name(th):
    th = wrap_deg(th)
    if abs(th)           < 0.5: return "+X"
    if abs(th - 90)      < 0.5: return "+Y"
    if abs(abs(th) - 180)< 0.5: return "-X"
    if abs(th + 90)      < 0.5: return "-Y"
    return f"{th:+.0f}°"

def parse_xy(s):
    s = s.strip().replace(",", " ")
    parts = [p for p in s.split() if p]
    if len(parts) != 2: return None
    try:    return clamp_world(float(parts[0]), float(parts[1]))
    except: return None

def parse_box_centre(s):
    """
    Parse  "N  x  y"  or  "N  x,y"  or  "N,x,y"  ->> (index_0based, x, y)
    Box number is 1-based in the UI, converted to 0-based here.
    Centre coordinates are NOT clamped -- user may intentionally place
    a box partially outside the map.
    """
    s = s.strip().replace(",", " ")
    parts = [p for p in s.split() if p]
    if len(parts) != 3:
        print(f"  BOX_CENTRE needs 3 values: box_num  x  y  (got {len(parts)}: {parts})")
        return None
    try:
        n   = int(parts[0]) - 1        # convert 1-based ->> 0-based
        cx  = float(parts[1])
        cy  = float(parts[2])
        return (n, cx, cy)
    except Exception as e:
        print(f"  Could not parse '{s}': {e}")
        return None

def parse_scale(s):
    """Parse a float scale value, clamped to [BOX_SCALE_MIN, BOX_SCALE_MAX]."""
    try:
        v = float(s.strip())
        return clamp(v, BOX_SCALE_MIN, BOX_SCALE_MAX)
    except:
        return None

# =====================================================================
# Main
# =====================================================================
def fit_boxes_to_map(boxes):
    """
    Compute a scale and per-box centre offset so that all boxes, when
    drawn together, fit inside the visible map with 20% margin.

    Returns (scale, [(cx_new, cy_new), ...])
    Each box keeps its own centre relative to the group centroid, just
    scaled down.  The whole group is then centred at (0, 0).

    If there is only one box the centroid IS that box's centre, so
    the scaled box ends up at (0, 0).
    """
    if not boxes:
        return 1.0, []

    # Group centroid
    gcx = sum(b["cx"] for b in boxes) / len(boxes)
    gcy = sum(b["cy"] for b in boxes) / len(boxes)

    # Compute the axis-aligned bounding box of ALL corners at scale=1
    all_corners = []
    for b in boxes:
        hw, hl = b["width"] / 2, b["length"] / 2
        rad = math.radians(b["rot_z_deg"])
        c, s = math.cos(rad), math.sin(rad)
        for lx, ly in [(hw,hl),(-hw,hl),(-hw,-hl),(hw,-hl)]:
            wx = b["cx"] + lx*c - ly*s - gcx   # relative to centroid
            wy = b["cy"] + lx*s + ly*c - gcy
            all_corners.append((wx, wy))

    if not all_corners:
        return 1.0, [(b["cx"], b["cy"]) for b in boxes]

    xs = [p[0] for p in all_corners]
    ys = [p[1] for p in all_corners]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)

    margin = 0.80   # use 80% of the map
    target_x = WORLD_W_M * margin
    target_y = WORLD_H_M * margin

    if span_x < 1e-6 and span_y < 1e-6:
        scale = 1.0
    elif span_x < 1e-6:
        scale = target_y / span_y
    elif span_y < 1e-6:
        scale = target_x / span_x
    else:
        scale = min(target_x / span_x, target_y / span_y)

    scale = round(clamp(scale, BOX_SCALE_MIN, BOX_SCALE_MAX), 4)

    # New centres: shift each box relative to group centroid, then scale offset
    new_centres = []
    for b in boxes:
        dx = (b["cx"] - gcx) * scale
        dy = (b["cy"] - gcy) * scale
        new_centres.append((round(dx, 4), round(dy, 4)))

    return scale, new_centres
def main():
    pygame.init(); pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("A* Navigator  3.0m × 1.2m")
    clock = pygame.time.Clock()

    # Serial
    ser = serial.Serial(PORT, BAUD, timeout=SER_TIMEOUT)
    time.sleep(0.2); ser.reset_input_buffer()
    send(ser, MOTOR_TYPE_CMD); time.sleep(0.05)
    send(ser, UPLOAD_ENABLE);  time.sleep(0.2)
    ser.reset_input_buffer()

    if wait_for_mall(ser, ENCODER_TIMEOUT_S) is None:
        print("No $MAll stream -- check cable/port.")
        ser.close(); pygame.quit(); return

    with SMBus(I2C_BUS) as bus:
        imu = IMUYaw(bus)
        imu.init()
        imu.calibrate_bias(BIAS_CAL_TIME)

        # ── State ──────────────────────────────────────────────────────
        x, y, th_cmd = 0.0, 0.0, 0.0
        imu.reset_pose(0.0)

        boxes:       list  = []                  # loaded obstacles
        circles:     list  = []                  # interactively placed circle obstacles
        occ:         list  = build_occupancy([]) # occupancy grid (empty)
        active_csv_path: str = ""                # path of currently loaded CSV
        goal_queue:  list  = []                  # ordered list of (x,y) goals to visit
        astar_path:  list  = []                  # smoothed A* path to current goal
        exp_world:   list  = []                  # explored cells for display
        exp_cache         = [None]               # cached explored surface
        waypoints:   list  = []                  # active drive queue (copy of astar_path)
        current_wp_idx    = 0                    # index into astar_path for display

        running_path = False
        paused       = False
        start_set_mode  = False
        circle_mode     = False   # left-click places a circle obstacle
        box_move_mode   = False   # left-click drags a CSV box centre
        drag_box_idx    = None    # index into boxes[] currently being dragged
        box_scale       = BOX_SCALE  # live scale for CSV box dimensions

        input_active = False
        input_mode   = None   # "GOAL" | "START" | "CSV"
        input_buf    = ""

        def set_start(wx, wy):
            nonlocal x, y
            x, y = round(wx, 3), round(wy, 3)
            imu.reset_pose(th_cmd)

        def plan_to_next():
            """
            Plan A* from current pose to goal_queue[0].
            Called automatically when a goal is reached or a new goal is queued.
            """
            nonlocal astar_path, exp_world, exp_cache, waypoints
            nonlocal running_path, current_wp_idx
            if not goal_queue:
                return
            gx, gy = goal_queue[0]
            print(f"Planning to goal {gx:+.3f},{gy:+.3f} "
                  f"(#{1} of {len(goal_queue)}) from ({x:.3f},{y:.3f}) ...")
            path, exp = plan_path((x, y), (gx, gy), occ)
            exp_world    = exp
            exp_cache[0] = None
            if path:
                astar_path     = path
                waypoints      = list(path)
                current_wp_idx = 0
                print(f"  Path ready: {len(path)} waypoints.  Press SPACE to run.")
            else:
                astar_path = []; waypoints = []
                print("  No path found to that goal.")

        def enqueue_goal(gx, gy):
            """
            Add (gx, gy) to the goal queue.
            If the queue was empty and the car isn't currently running, plan immediately.
            If the car is already running (mid-path), just append -- the next goal will
            be planned automatically when the current one is reached.
            """
            nonlocal running_path
            pt = (round(gx, 3), round(gy, 3))
            goal_queue.append(pt)
            print(f"  Goal #{len(goal_queue)} queued: {pt}")
            # Plan immediately only if this is the first goal and we're idle
            if len(goal_queue) == 1 and not running_path:
                plan_to_next()

        def rebuild_occ():
            """Rebuild occupancy grid from current boxes + circles + scale."""
            nonlocal occ, exp_cache
            occ = build_occupancy(boxes, circles, box_scale)
            exp_cache[0] = None

        def reload_boxes(path_str):
            nonlocal boxes, active_csv_path
            boxes = load_boxes(path_str)
            active_csv_path = path_str
            rebuild_occ()
            print(f"  Occupancy grid rebuilt ({GRID_RES*100:.0f}cm resolution).")

        def add_circle(cx, cy):
            """Place a circle obstacle, rebuild grid, replan if goal exists."""
            cx, cy = round(cx, 3), round(cy, 3)
            circles.append((cx, cy))
            rebuild_occ()
            print(f"  Circle added at ({cx:+.3f},{cy:+.3f}). Total: {len(circles)}")
            if goal_queue:
                print("  Replanning around new obstacle ...")
                plan_to_next()

        def remove_last_circle():
            """Undo the last placed circle, rebuild grid, replan."""
            if not circles:
                return
            removed = circles.pop()
            rebuild_occ()
            print(f"  Removed circle at ({removed[0]:+.3f},{removed[1]:+.3f}). "
                  f"Remaining: {len(circles)}")
            if goal_queue:
                plan_to_next()

        # Auto-load the default CSV on startup
        if BBOX_CSV_PATH:
            reload_boxes(BBOX_CSV_PATH)

        # Print controls
        print("\n=== A* Navigator ===")
        print("Left-click   : queue goal (normal) | drag box (B) | place circle (O) | set start (S)")
        print("Right-click  : remove last queued goal (normal) | undo circle (O)")
        print("G            : type goal x,y  -- adds to queue")
        print("F            : fit all CSV boxes into the visible map automatically")
        print("M            : type box centre  -- format:  box_num  x  y   e.g.  1  0.5  0.1")
        print("K            : type exact box scale value  e.g.  0.25")
        print("B            : toggle box-move mode  (drag CSV box centres with mouse)")
        print("[ / ]        : decrease / increase CSV box scale by 5% steps")
        print("T            : type start x,y")
        print("O            : toggle circle-obstacle placement mode")
        print("N            : type circle centre x,y")
        print("Z            : clear all circle obstacles")
        print("L            : load obstacle CSV")
        print("R            : replan current leg from current position")
        print("SPACE        : start / resume driving queue")
        print("P            : pause")
        print("C            : clear entire goal queue + stop")
        print("1/2/3/4      : set facing +X/+Y/-X/-Y")
        print("S            : toggle set-start mode")
        print("ESC          : quit\n")

        try:
            while True:
                # ── Events ────────────────────────────────────────────
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt

                    # ── Typed input ───────────────────────────────────
                    if event.type == pygame.KEYDOWN and input_active:
                        if event.key == pygame.K_ESCAPE:
                            input_active = False; input_mode = None; input_buf = ""

                        elif event.key == pygame.K_BACKSPACE:
                            input_buf = input_buf[:-1]

                        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            if input_mode == "CSV":
                                p = input_buf.strip()
                                if p: reload_boxes(p)
                            elif input_mode == "GOAL":
                                xy = parse_xy(input_buf)
                                if xy: enqueue_goal(*xy)
                            elif input_mode == "START":
                                xy = parse_xy(input_buf)
                                if xy: set_start(*xy)
                            elif input_mode == "CIRCLE":
                                xy = parse_xy(input_buf)
                                if xy: add_circle(*xy)
                            elif input_mode == "BOX_CENTRE":
                                result = parse_box_centre(input_buf)
                                if result:
                                    idx, cx, cy = result
                                    if 0 <= idx < len(boxes):
                                        old = (boxes[idx]["cx"], boxes[idx]["cy"])
                                        boxes[idx]["cx"] = cx
                                        boxes[idx]["cy"] = cy
                                        rebuild_occ()
                                        print(f"  Box {idx+1} centre: "
                                              f"({old[0]:+.3f},{old[1]:+.3f}) ->> "
                                              f"({cx:+.3f},{cy:+.3f})")
                                        if goal_queue: plan_to_next()
                                    else:
                                        print(f"  Box {idx+1} doesn't exist "
                                              f"(loaded: {len(boxes)})")
                                else:
                                    print("  Format: box_num  x  y   e.g.  1  0.5  0.1")
                            elif input_mode == "BOX_SCALE":
                                v = parse_scale(input_buf)
                                if v is not None:
                                    box_scale = v
                                    rebuild_occ()
                                    print(f"  Box scale set to {box_scale:.4f}  "
                                          f"({box_scale*100:.1f}% of CSV size)")
                                    if goal_queue: plan_to_next()
                                else:
                                    print(f"  Enter a number between "
                                          f"{BOX_SCALE_MIN} and {BOX_SCALE_MAX}")
                            input_active = False; input_mode = None; input_buf = ""

                        else:
                            ch = event.unicode
                            if input_mode == "CSV":
                                if ch and ch not in ("\n", "\r"): input_buf += ch
                            elif input_mode in ("BOX_CENTRE", "BOX_SCALE"):
                                # Allow digits, sign, decimal, comma, space
                                if ch and (ch.isdigit() or ch in "+-., "):
                                    input_buf += ch
                            elif ch and (ch.isdigit() or ch in "+-., "):
                                input_buf += ch
                        continue

                    # ── Normal keyboard ───────────────────────────────
                    if event.type == pygame.KEYDOWN:
                        k = event.key

                        if k == pygame.K_ESCAPE:
                            raise KeyboardInterrupt

                        elif k == pygame.K_SPACE:
                            if astar_path:
                                running_path = True; paused = False
                            elif goal_queue:
                                plan_to_next()
                                running_path = True; paused = False
                            else:
                                print("No goals queued -- click to add goals first.")

                        elif k == pygame.K_p:
                            paused = not paused
                            if paused: stop_motors(ser)

                        elif k == pygame.K_c:
                            # Clear entire goal queue, path and stop
                            goal_queue.clear()
                            waypoints.clear(); astar_path = []
                            running_path = False; paused = False
                            exp_world = []; exp_cache[0] = None
                            stop_motors(ser)
                            print("  Goal queue cleared.")

                        elif k == pygame.K_r:
                            # Replan current leg from current position
                            if goal_queue:
                                print("Replanning current leg ...")
                                stop_motors(ser)
                                running_path = False
                                plan_to_next()
                            else:
                                print("No goals queued.")

                        elif k == pygame.K_g:
                            input_active = True; input_mode = "GOAL"; input_buf = ""

                        elif k == pygame.K_t:
                            input_active = True; input_mode = "START"; input_buf = ""

                        elif k == pygame.K_l:
                            input_active = True; input_mode = "CSV"
                            # Pre-fill with current path so user can edit rather than retype
                            input_buf = active_csv_path

                        elif k == pygame.K_s:
                            start_set_mode = not start_set_mode
                            if start_set_mode: circle_mode = False; box_move_mode = False

                        elif k == pygame.K_b:
                            # Toggle box-move mode -- click near a box centre to drag it
                            box_move_mode = not box_move_mode
                            if box_move_mode: circle_mode = False; start_set_mode = False
                            drag_box_idx = None

                        elif k == pygame.K_o:
                            circle_mode    = not circle_mode
                            if circle_mode: start_set_mode = False; box_move_mode = False

                        elif k == pygame.K_LEFTBRACKET:
                            # Decrease box scale
                            box_scale = round(max(BOX_SCALE_MIN,
                                                  box_scale - BOX_SCALE_STEP), 4)
                            rebuild_occ()
                            print(f"  Box scale ->> {box_scale:.2f}  "
                                  f"(boxes at {box_scale*100:.0f}% of CSV size)")
                            if goal_queue: plan_to_next()

                        elif k == pygame.K_RIGHTBRACKET:
                            # Increase box scale
                            box_scale = round(min(BOX_SCALE_MAX,
                                                  box_scale + BOX_SCALE_STEP), 4)
                            rebuild_occ()
                            print(f"  Box scale ->> {box_scale:.2f}  "
                                  f"(boxes at {box_scale*100:.0f}% of CSV size)")
                            if goal_queue: plan_to_next()

                        elif k == pygame.K_n:
                            # Type a circle centre coordinate
                            input_active = True; input_mode = "CIRCLE"; input_buf = ""

                        elif k == pygame.K_m:
                            # Type exact box centre: "box_num  x  y"
                            if boxes:
                                input_active = True; input_mode = "BOX_CENTRE"
                                input_buf = ""
                            else:
                                print("  No boxes loaded -- press L to load a CSV first.")

                        elif k == pygame.K_k:
                            # Type exact box scale value
                            input_active = True; input_mode = "BOX_SCALE"
                            input_buf = str(box_scale)

                        elif k == pygame.K_f:
                            # Fit all CSV boxes into the visible map automatically
                            if boxes:
                                new_scale, new_centres = fit_boxes_to_map(boxes)
                                for i, (cx, cy) in enumerate(new_centres):
                                    boxes[i]["cx"] = cx
                                    boxes[i]["cy"] = cy
                                box_scale = new_scale
                                rebuild_occ()
                                print(f"  Fit to map: scale={box_scale:.4f}  "
                                      f"({box_scale*100:.1f}% of CSV size)")
                                for i, b in enumerate(boxes):
                                    print(f"    Box {i+1}: centre ->> "
                                          f"({b['cx']:+.4f}, {b['cy']:+.4f})  "
                                          f"scaled size: "
                                          f"{b['width']*box_scale*100:.1f}×"
                                          f"{b['length']*box_scale*100:.1f} cm")
                                if goal_queue: plan_to_next()
                            else:
                                print("  No boxes loaded.")

                        elif k == pygame.K_z:
                            # Clear all interactively placed circles
                            circles.clear()
                            rebuild_occ()
                            print("  All circle obstacles cleared.")
                            if goal_queue: plan_to_next()

                        elif k == pygame.K_1: th_cmd = 0.0;   imu.reset_pose(th_cmd)
                        elif k == pygame.K_2: th_cmd = 90.0;  imu.reset_pose(th_cmd)
                        elif k == pygame.K_3: th_cmd = 180.0; imu.reset_pose(th_cmd)
                        elif k == pygame.K_4: th_cmd = -90.0; imu.reset_pose(th_cmd)

                    # ── Mouse click ───────────────────────────────────
                    if (not input_active) and event.type == pygame.MOUSEBUTTONDOWN:
                        w = from_pygame(pygame.mouse.get_pos())
                        if event.button == 1:
                            if start_set_mode:
                                set_start(*w)
                            elif circle_mode:
                                add_circle(*w)
                            elif box_move_mode and boxes:
                                # Find closest box centre within grab radius
                                GRAB_PX = 20
                                best_i, best_d = None, GRAB_PX + 1
                                for i, b in enumerate(boxes):
                                    bpx, bpy = to_pygame((b["cx"], b["cy"]))
                                    mpx, mpy = pygame.mouse.get_pos()
                                    d = math.hypot(bpx - mpx, bpy - mpy)
                                    if d < best_d:
                                        best_d = d; best_i = i
                                drag_box_idx = best_i
                                if drag_box_idx is not None:
                                    print(f"  Grabbing box {drag_box_idx+1} "
                                          f"at ({boxes[drag_box_idx]['cx']:+.3f},"
                                          f"{boxes[drag_box_idx]['cy']:+.3f})")
                            else:
                                # Normal left-click ->> add to goal queue
                                enqueue_goal(*w)
                        elif event.button == 3:
                            if circle_mode:
                                remove_last_circle()
                            else:
                                # Right-click in normal mode ->> remove last queued goal
                                if goal_queue:
                                    removed = goal_queue.pop()
                                    print(f"  Removed last goal {removed}. "
                                          f"Queue: {len(goal_queue)} remaining.")
                                    if not running_path and goal_queue:
                                        plan_to_next()
                                    elif not goal_queue:
                                        waypoints.clear(); astar_path = []
                                        exp_world = []; exp_cache[0] = None

                    # ── Mouse drag (box repositioning) ────────────────
                    if (not input_active) and event.type == pygame.MOUSEMOTION:
                        if box_move_mode and drag_box_idx is not None:
                            wx, wy = from_pygame(pygame.mouse.get_pos())
                            boxes[drag_box_idx]["cx"] = wx
                            boxes[drag_box_idx]["cy"] = wy
                            rebuild_occ()

                    # ── Mouse release ─────────────────────────────────
                    if (not input_active) and event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 1 and drag_box_idx is not None:
                            b = boxes[drag_box_idx]
                            print(f"  Box {drag_box_idx+1} moved to "
                                  f"({b['cx']:+.3f}, {b['cy']:+.3f})")
                            drag_box_idx = None
                            rebuild_occ()
                            if goal_queue:
                                plan_to_next()

                # ── Drawing ───────────────────────────────────────────
                draw_grid(screen)
                draw_boxes(screen, boxes, box_scale, drag_box_idx, box_move_mode)
                draw_circle_obstacles(screen, circles, circle_mode)

                # Explored cells (faint, cached after first render)
                if exp_world:
                    draw_explored(screen, exp_world, exp_cache)

                # A* path
                if astar_path:
                    draw_path(screen, astar_path, current_wp_idx)

                # Goal queue markers
                draw_goal_queue(screen, goal_queue)

                # Robot
                yaw_now = imu.update()
                draw_robot(screen, x, y, yaw_now)

                # HUD
                mode_str = ("SET-START"       if start_set_mode else
                            "PLACE-CIRCLE"    if circle_mode    else
                            "MOVE-BOX"        if box_move_mode  else
                            "SET-GOAL (click)")
                draw_text(screen,
                    f"A* Navigator | Car {CAR_LENGTH_M*100:.0f}×{CAR_WIDTH_M*100:.0f}cm"
                    f" | CSV boxes={len(boxes)} | Circles={len(circles)} | Mode={mode_str}",
                    10, 10, DARK)
                draw_text(screen,
                    f"Pos: ({x:+.2f},{y:+.2f}) Facing={heading_name(th_cmd)}"
                    f"  Goals queued: {len(goal_queue)}"
                    f"{'  Next: ' + f'({goal_queue[0][0]:+.2f},{goal_queue[0][1]:+.2f})' if goal_queue else ''}",
                    10, 32, DARK)
                draw_text(screen,
                    f"IMU={yaw_now:+.1f}° hdg_err={wrap_deg(th_cmd-yaw_now):+.1f}° | "
                    f"WPs remaining={len(waypoints)} | "
                    f"Running={running_path}  Paused={paused}",
                    10, 54, DARK)
                csv_display = active_csv_path if active_csv_path else "none"
                draw_text(screen,
                    f"CSV: {csv_display}  |  Box scale: {box_scale:.2f}×"
                    f"  ({box_scale*100:.0f}% of CSV size)  [ / ] to adjust",
                    10, 74, DARK, 15)
                draw_text(screen,
                    "click=add-goal | F fit-boxes | B box-move | M box-centre | K box-scale | [ ] scale-step | "
                    "R-click=rm-goal | G goal | O circle | N circle-xy | Z clr-circles | "
                    "S start | T start-xy | L csv | R replan | 1-4 face | SPACE run | P pause | C clear | ESC",
                    10, HEIGHT - 28, DARK, 12)

                if input_active:
                    draw_input_overlay(screen, input_mode, input_buf,
                                       active_csv_path, boxes, box_scale)

                pygame.display.flip()
                clock.tick(30)

                # ── Execute next waypoint leg ─────────────────────────
                if running_path and not paused and not input_active and waypoints:
                    tx, ty = waypoints[0]
                    dx, dy = tx - x, ty - y
                    dist_m = math.hypot(dx, dy)

                    if dist_m < WP_ARRIVE_M:
                        waypoints.pop(0)
                        # Advance display index to highlight next waypoint
                        current_wp_idx = max(0,
                            len(astar_path) - len(waypoints) - 1)
                        stop_motors(ser)
                        continue

                    desired_heading = math.degrees(math.atan2(dy, dx))
                    dtheta = wrap_deg(desired_heading - th_cmd)

                    # Turn to face the next waypoint
                    do_turn_to_heading_imu(ser, imu, dtheta)

                    # Brief settle pause -- lets IMU gyro stop ringing before reading
                    time.sleep(TURN_SETTLE_S)

                    # Read the actual yaw the IMU measured after the turn.
                    # Using the commanded heading (th_cmd + dtheta) here silently
                    # carries turn error into every subsequent leg.  Using the
                    # real measured yaw means each leg starts from truth.
                    actual_yaw = imu.update()
                    turn_err   = wrap_deg(desired_heading - actual_yaw)
                    print(f"  Turn done -- commanded={desired_heading:+.1f}°  "
                          f"actual={actual_yaw:+.1f}°  err={turn_err:+.1f}°")

                    # Accept actual yaw as the new heading reference
                    th_cmd = actual_yaw
                    imu.reset_pose(actual_yaw)

                    # Drive the leg with PI heading hold using actual heading
                    do_drive_distance_with_imu(ser, imu, dist_m,
                                               target_heading_deg=th_cmd)

                    # Update dead-reckoning pose using actual heading so
                    # position error does not compound across legs
                    x += dist_m * math.cos(math.radians(th_cmd))
                    y += dist_m * math.sin(math.radians(th_cmd))
                    x, y = clamp_world(x, y)

                elif running_path and not waypoints:
                    # Current goal reached -- pop it and move to the next
                    completed = goal_queue.pop(0)
                    stop_motors(ser)
                    print(f"Goal {completed} reached. "
                          f"Pose: ({x:.3f},{y:.3f}). "
                          f"{len(goal_queue)} goal(s) remaining.")
                    if goal_queue:
                        # Plan and immediately start driving to the next goal
                        plan_to_next()
                        if astar_path:
                            running_path = True
                    else:
                        running_path = False
                        print("All goals completed.")

        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            try:
                stop_motors(ser)
                if ser.is_open:
                    send(ser, UPLOAD_DISABLE)
                    ser.close()
            except Exception as e:
                print(f"Cleanup error: {e}")
            pygame.quit()


if __name__ == "__main__":
    main()
