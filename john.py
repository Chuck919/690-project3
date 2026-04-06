from controller import Robot, Camera, Lidar
import math

TIME_STEP = 64
MAX_SPEED = 10

robot = Robot()

# ------------------ CAMERA ------------------
camera = robot.getDevice('camera')
camera.enable(TIME_STEP)
cam_width = camera.getWidth()
cam_height = camera.getHeight()

# ------------------ LIDAR ------------------
lidar = robot.getDevice('LDS-01')
lidar.enable(TIME_STEP)

lidar_resolution = lidar.getHorizontalResolution()
lidar_fov = lidar.getFov()

# ------------------ WHEELS ------------------
wheels = []
wheel_names = [
    'front_left_wheel',
    'front_right_wheel',
    'back_left_wheel',
    'back_right_wheel'
]
for name in wheel_names:
    w = robot.getDevice(name)
    w.setPosition(float('inf'))
    w.setVelocity(0.0)
    wheels.append(w)

def set_speed(left, right):
    wheels[0].setVelocity(left)   # front left
    wheels[2].setVelocity(left)   # back left
    wheels[1].setVelocity(right)  # front right
    wheels[3].setVelocity(right)  # back right

# ------------------ DISTANCE SENSORS ------------------
ds_names = ['ds_front_left', 'ds_front_right', 'ds_back', 'ds_right', 'ds_left']
distance_sensors = {}
for name in ds_names:
    ds = robot.getDevice(name)
    ds.enable(TIME_STEP)
    distance_sensors[name] = ds

def get_ds(name):
    return distance_sensors[name].getValue()

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ------------------ COLOR DETECTION ------------------
def is_yellow(r, g, b):
    if r > 140 and g > 140 and b < 100:
        return True
    if r > 160 and g > 160 and b < 130 and r > b + 50 and g > b + 50:
        return True
    if r > 100 and g > 100 and b < 60 and r > b + 60 and g > b + 60:
        return True
    return False

def is_cyan(r, g, b):
    return g > 120 and b > 120 and r < 100 and abs(g - b) < 80

def is_magenta(r, g, b):
    return r > 120 and b > 120 and g < 100 and abs(r - b) < 80

# ------------------ LIDAR HELPERS ------------------
def get_lidar_distances():
    return list(lidar.getRangeImage())

def get_lidar_sector(distances, sector='front'):
    n = len(distances)
    sectors = {
        'left':        distances[0           : n // 5],
        'front-left':  distances[n // 5      : 2 * n // 5],
        'front':       distances[2 * n // 5  : 3 * n // 5],
        'front-right': distances[3 * n // 5  : 4 * n // 5],
        'right':       distances[4 * n // 5  :],
    }
    readings = sectors.get(sector, [])
    return min((d for d in readings if not math.isinf(d)), default=float('inf'))

# ------------------ STATE ------------------
opponent_goal = None   # 'cyan' or 'magenta'
startup_frames = 0
last_ball_dir = 1      # 1=right, -1=left — for search spin direction
last_opp_dir = 1       # 1=right, -1=left — last camera-x side we saw opponent goal
lost_timer = 0
stuck_timer = 0
prev_yellow_count = 0

# ------------------ MAIN LOOP ------------------
while robot.step(TIME_STEP) != -1:
    image = camera.getImage()

    # === SCAN CAMERA FOR COLORS ===
    yellow_x_sum = 0
    yellow_y_sum = 0
    yellow_count = 0
    cyan_count = 0
    cyan_x_sum = 0
    magenta_count = 0
    magenta_x_sum = 0

    for y in range(cam_height):
        for x in range(cam_width):
            r = Camera.imageGetRed(image, cam_width, x, y)
            g = Camera.imageGetGreen(image, cam_width, x, y)
            b = Camera.imageGetBlue(image, cam_width, x, y)

            if is_yellow(r, g, b):
                yellow_x_sum += x
                yellow_y_sum += y
                yellow_count += 1
            elif is_cyan(r, g, b):
                cyan_x_sum += x
                cyan_count += 1
            elif is_magenta(r, g, b):
                magenta_x_sum += x
                magenta_count += 1

    # === STARTUP: DETECT OPPONENT GOAL ===
    if opponent_goal is None:
        startup_frames += 1
        print(f"[STARTUP] frame={startup_frames} cyan={cyan_count} magenta={magenta_count}")
        if cyan_count > 15:
            opponent_goal = 'cyan'
            print(f"[STARTUP] Opponent goal = CYAN")
        elif magenta_count > 15:
            opponent_goal = 'magenta'
            print(f"[STARTUP] Opponent goal = MAGENTA")
        elif startup_frames > 8:
            opponent_goal = 'magenta'
            print(f"[STARTUP] Timeout — defaulting opponent goal = MAGENTA")
        else:
            set_speed(0, 0)
            continue

    # === BALL INFO ===
    ball_seen = yellow_count > 3
    if ball_seen:
        ball_x = yellow_x_sum / yellow_count
        ball_norm_x = (ball_x / cam_width - 0.5) * 2   # -1 to +1
        last_ball_dir = 1 if ball_norm_x >= 0 else -1
    else:
        ball_norm_x = 0

    # === GOAL VISIBILITY ===
    goal_threshold = 20
    if opponent_goal == 'cyan':
        see_opponent = cyan_count > goal_threshold
        see_own = magenta_count > goal_threshold
        opp_x_sum, opp_count = cyan_x_sum, cyan_count
    else:
        see_opponent = magenta_count > goal_threshold
        see_own = cyan_count > goal_threshold
        opp_x_sum, opp_count = magenta_x_sum, magenta_count

    # Track where opponent goal was last seen in camera
    if see_opponent and opp_count > 0:
        opp_avg_x = opp_x_sum / opp_count
        last_opp_dir = 1 if opp_avg_x > cam_width / 2 else -1

    # === LIDAR ===
    distances = get_lidar_distances()
    front_dist = get_lidar_sector(distances, 'front')
    left_dist = get_lidar_sector(distances, 'left')
    right_dist = get_lidar_sector(distances, 'right')

    # === STUCK DETECTION ===
    # If charging at ball for a long time but not getting closer, back up
    if ball_seen and yellow_count > 30:
        if yellow_count <= prev_yellow_count + 2:
            stuck_timer += 1
        else:
            stuck_timer = 0
    else:
        stuck_timer = 0
    prev_yellow_count = yellow_count

    if stuck_timer > 40:
        # Stuck against wall or opponent — back up and turn
        print(f"[STUCK] Backing up (yellow={yellow_count}, front={front_dist:.2f})")
        set_speed(-MAX_SPEED * 0.6, -MAX_SPEED * 0.3)
        stuck_timer = 0
        continue

    # ===========================================================
    #  BEHAVIOR: only two states — SEARCH or CHARGE
    # ===========================================================

    if not ball_seen:
        # ---- SEARCH: drive forward while sweeping to find ball ----
        lost_timer += 1
        if lost_timer > 60:
            last_ball_dir *= -1
            lost_timer = 0

        if front_dist < 0.2:
            print(f"[SEARCH] wall ({front_dist:.2f}m), turning")
            set_speed(MAX_SPEED * last_ball_dir, -MAX_SPEED * last_ball_dir)
        else:
            print(f"[SEARCH] timer={lost_timer} dir={last_ball_dir} yellow={yellow_count} front={front_dist:.2f}")
            set_speed(MAX_SPEED * 0.6 + MAX_SPEED * 0.4 * last_ball_dir,
                      MAX_SPEED * 0.6 - MAX_SPEED * 0.4 * last_ball_dir)

    else:
        # ---- CHARGE: ball visible — always go for it at max speed ----
        lost_timer = 0

        # If we can also see the opponent goal, adjust aim to push ball goalward
        aim_adj = 0.0
        if see_opponent and opp_count > 0:
            opp_norm_x = ((opp_x_sum / opp_count) / cam_width - 0.5) * 2
            goal_offset = opp_norm_x - ball_norm_x
            aim_adj = clamp(-goal_offset * 0.25, -0.3, 0.3)

        target_x = ball_norm_x + aim_adj
        steer = target_x * 6.0

        l = clamp(MAX_SPEED + steer, -MAX_SPEED, MAX_SPEED)
        r = clamp(MAX_SPEED - steer, -MAX_SPEED, MAX_SPEED)
        tag = "CHARGE" if see_opponent else "CHASE"
        print(f"[{tag}] ball_nx={ball_norm_x:.2f} aim_adj={aim_adj:.2f} L={l:.1f} R={r:.1f} yellow={yellow_count}")
        set_speed(l, r)
