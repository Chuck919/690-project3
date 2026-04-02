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

def is_purple(r, g, b):
    if r > 100 and g < 80 and b > 100 and r > g + 30 and b > g + 30:
        return True
    if r > 80 and g < 60 and b > 80:
        return True
    return False

# ------------------ LIDAR ------------------
def get_lidar_distances():
    return list(lidar.getRangeImage())

def get_lidar_sector(distances, sector='front'):
    n = len(distances)
    sectors = {
        'left':        distances[0          : n // 5],
        'front-left':  distances[n // 5     : 2 * n // 5],
        'front':       distances[2 * n // 5 : 3 * n // 5],
        'front-right': distances[3 * n // 5 : 4 * n // 5],
        'right':       distances[4 * n // 5 :],
    }
    readings = sectors.get(sector, [])
    return min((d for d in readings if not math.isinf(d)), default=float('inf'))

# ------------------ STATE ------------------
stuck_timer        = 0
stuck_escape_timer = 0
last_ball_x        = None

# ------------------ MAIN LOOP ------------------
while robot.step(TIME_STEP) != -1:
    leftSpeed = 0.0
    rightSpeed = 0.0

    # ================= LIDAR =================
    distances = get_lidar_distances()

    # Sanity check — if readings are mostly invalid, physics engine
    # is struggling this step, stop and wait for it to recover
    valid_readings = [d for d in distances if not math.isinf(d) and d > 0]
    if len(valid_readings) < 5:
        set_speed(0, 0)
        continue

    # ================= CAMERA SCAN =================
    image = camera.getImage()

    yellow_x_sum = 0
    yellow_count = 0
    yellow_y_sum = 0

    purple_x_sum = 0
    purple_count = 0

    for y in range(cam_height):
        for x in range(cam_width):
            r = Camera.imageGetRed(image, cam_width, x, y)
            g = Camera.imageGetGreen(image, cam_width, x, y)
            b = Camera.imageGetBlue(image, cam_width, x, y)
            if is_yellow(r, g, b):
                yellow_x_sum += x
                yellow_y_sum += y
                yellow_count += 1
            if is_purple(r, g, b):
                purple_x_sum += x
                purple_count += 1

    # ================= STUCK DETECTION =================
    if yellow_count > 5:
        ball_x = yellow_x_sum / yellow_count
        ball_y = yellow_y_sum / yellow_count

        # If ball hasn't moved much in camera for many frames, we're probably pinned
        if last_ball_x is not None and abs(ball_x - last_ball_x) < 3:
            stuck_timer += 1
        else:
            stuck_timer = 0
        last_ball_x = ball_x

        # Trigger escape if stuck for ~2 seconds (30 frames)
        if stuck_timer > 30:
            stuck_escape_timer = 25
            stuck_timer = 0
    else:
        # Lost sight of ball — reset stuck tracking
        stuck_timer = 0
        last_ball_x = None

    # ================= BEHAVIOR =================

    if stuck_escape_timer > 0:
        # ---- ESCAPE MANEUVER ----
        # Back up and turn to unpin from wall
        leftSpeed          = -MAX_SPEED
        rightSpeed         = -0.3 * MAX_SPEED
        stuck_escape_timer -= 1

    elif yellow_count > 5:
        # ---- CAN SEE BALL ----
        ball_offset = (ball_x - cam_width / 2) / (cam_width / 2)

        if purple_count > 5:
            # Can see both ball and goal
            goal_x = purple_x_sum / purple_count

            if goal_x > cam_width / 2:
                # Goal is to the right — approach ball from the left
                if ball_offset < -0.2:
                    leftSpeed  = MAX_SPEED
                    rightSpeed = 0.5 * MAX_SPEED
                elif ball_offset > 0.2:
                    leftSpeed  = 0.5 * MAX_SPEED
                    rightSpeed = MAX_SPEED
                else:
                    leftSpeed  = MAX_SPEED
                    rightSpeed = MAX_SPEED
            else:
                # Goal is to the left — approach ball from the right
                if ball_offset > 0.2:
                    leftSpeed  = 0.5 * MAX_SPEED
                    rightSpeed = MAX_SPEED
                elif ball_offset < -0.2:
                    leftSpeed  = MAX_SPEED
                    rightSpeed = 0.5 * MAX_SPEED
                else:
                    leftSpeed  = MAX_SPEED
                    rightSpeed = MAX_SPEED

        else:
            # Can see ball but not goal — drive toward ball
            if ball_offset < -0.3:
                leftSpeed  = 0.4 * MAX_SPEED
                rightSpeed = MAX_SPEED
            elif ball_offset > 0.3:
                leftSpeed  = MAX_SPEED
                rightSpeed = 0.4 * MAX_SPEED
            else:
                leftSpeed  = MAX_SPEED
                rightSpeed = MAX_SPEED

    else:
        # ---- CANNOT SEE BALL ----
        # Spin in place to search
        leftSpeed  =  0.5 * MAX_SPEED
        rightSpeed = -0.5 * MAX_SPEED

    set_speed(leftSpeed, rightSpeed)
