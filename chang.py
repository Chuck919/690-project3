from controller import Robot, Camera, Lidar
import math

TIME_STEP = 64
MAX_SPEED = 10.0

SEARCH_TURN_SPEED = 6.5
APPROACH_SPEED = 8.8
CHARGE_SPEED = 10.0
ESCAPE_REVERSE_SPEED = 7.5
ESCAPE_TURN_SPEED = 6.5

BALL_PIXEL_MIN = 6
BALL_CLOSE_PIXELS = 110
BALL_CENTER_TOLERANCE = 0.16
GOAL_PIXEL_MIN = 65
GOAL_TRAP_STEPS = 18
BALL_MEMORY_STEPS = 16
OWN_GOAL_DANGER_ALIGNMENT = 0.18
OWN_GOAL_BIG_COUNT = 220
BALL_STRONG_PIXELS = 22
GOAL_ESCAPE_COUNT = 120

robot = Robot()

camera = robot.getDevice("camera")
camera.enable(TIME_STEP)
cam_width = camera.getWidth()
cam_height = camera.getHeight()

lidar = robot.getDevice("LDS-01")
lidar.enable(TIME_STEP)

wheels = []
for name in [
    "front_left_wheel",
    "front_right_wheel",
    "back_left_wheel",
    "back_right_wheel",
]:
    wheel = robot.getDevice(name)
    wheel.setPosition(float("inf"))
    wheel.setVelocity(0.0)
    wheels.append(wheel)

distance_sensors = {}
for name in ["ds_front_left", "ds_front_right", "ds_back", "ds_right", "ds_left"]:
    sensor = robot.getDevice(name)
    sensor.enable(TIME_STEP)
    distance_sensors[name] = sensor


def clamp(value, low, high):
    return max(low, min(high, value))


def set_speed(left, right):
    left = clamp(left, -MAX_SPEED, MAX_SPEED)
    right = clamp(right, -MAX_SPEED, MAX_SPEED)
    wheels[0].setVelocity(left)
    wheels[2].setVelocity(left)
    wheels[1].setVelocity(right)
    wheels[3].setVelocity(right)


def get_ds(name):
    return distance_sensors[name].getValue()


def is_yellow(r, g, b):
    return (
        (r > 140 and g > 140 and b < 100)
        or (r > 160 and g > 160 and b < 130 and r > b + 50 and g > b + 50)
        or (r > 100 and g > 100 and b < 60 and r > b + 60 and g > b + 60)
    )


def is_cyan(r, g, b):
    return (
        (g > 120 and b > 120 and r < 100)
        or (g > 150 and b > 150 and r < 140 and g > r + 40 and b > r + 40)
    )


def is_magenta(r, g, b):
    return (
        (r > 120 and b > 120 and g < 100)
        or (r > 150 and b > 150 and g < 140 and r > g + 40 and b > g + 40)
    )


def get_lidar_sector(distances, sector="front"):
    count = len(distances)
    sectors = {
        "left": distances[0 : count // 5],
        "front-left": distances[count // 5 : 2 * count // 5],
        "front": distances[2 * count // 5 : 3 * count // 5],
        "front-right": distances[3 * count // 5 : 4 * count // 5],
        "right": distances[4 * count // 5 :],
    }
    readings = sectors.get(sector, [])
    return min((value for value in readings if not math.isinf(value)), default=float("inf"))


def detect_blob(image, predicate, x_step=2, y_step=2):
    pixel_count = 0
    x_sum = 0
    lowest_y = -1

    for y in range(0, cam_height, y_step):
        for x in range(0, cam_width, x_step):
            r = Camera.imageGetRed(image, cam_width, x, y)
            g = Camera.imageGetGreen(image, cam_width, x, y)
            b = Camera.imageGetBlue(image, cam_width, x, y)
            if predicate(r, g, b):
                pixel_count += 1
                x_sum += x
                if y > lowest_y:
                    lowest_y = y

    if pixel_count == 0:
        return {"count": 0, "cx": cam_width / 2.0, "lowest_y": -1}

    return {
        "count": pixel_count,
        "cx": x_sum / pixel_count,
        "lowest_y": lowest_y,
    }


def detect_ball_target(image):
    pixel_count = 0
    weighted_x_sum = 0.0
    weight_sum = 0.0
    lowest_y = -1

    for y in range(0, cam_height, 1):
        for x in range(0, cam_width, 1):
            r = Camera.imageGetRed(image, cam_width, x, y)
            g = Camera.imageGetGreen(image, cam_width, x, y)
            b = Camera.imageGetBlue(image, cam_width, x, y)
            if is_yellow(r, g, b):
                pixel_count += 1
                weight = 1.0 + (y / cam_height) * 2.5
                weighted_x_sum += x * weight
                weight_sum += weight
                if y > lowest_y:
                    lowest_y = y

    if pixel_count == 0:
        return {"count": 0, "cx": cam_width / 2.0, "lowest_y": -1}

    return {
        "count": pixel_count,
        "cx": weighted_x_sum / weight_sum,
        "lowest_y": lowest_y,
    }


def steer_to_x(target_x, drive_speed, turn_gain=5.5):
    center_x = cam_width / 2.0
    error = (target_x - center_x) / center_x
    turn = clamp(error * turn_gain, -5.5, 5.5)
    return drive_speed + turn, drive_speed - turn


robot_name = robot.getName()
own_goal = "cyan" if robot_name == "robot(1)" else "magenta"
opponent_goal = "magenta" if own_goal == "cyan" else "cyan"
search_direction = 1 if robot_name == "robot(1)" else -1
escape_timer = 0
escape_direction = search_direction
last_ball_x = cam_width / 2.0
ball_memory_timer = 0


while robot.step(TIME_STEP) != -1:
    image = camera.getImage()
    ball = detect_ball_target(image)
    own_goal_view = detect_blob(
        image, is_cyan if own_goal == "cyan" else is_magenta, x_step=3, y_step=3
    )
    opponent_goal_view = detect_blob(
        image, is_cyan if opponent_goal == "cyan" else is_magenta, x_step=3, y_step=3
    )

    distances = list(lidar.getRangeImage())
    front_dist = get_lidar_sector(distances, "front")
    left_dist = get_lidar_sector(distances, "left")
    right_dist = get_lidar_sector(distances, "right")

    ds_front_left = get_ds("ds_front_left")
    ds_front_right = get_ds("ds_front_right")
    ds_back = get_ds("ds_back")
    ds_left = get_ds("ds_left")
    ds_right = get_ds("ds_right")

    ball_visible = ball["count"] >= BALL_PIXEL_MIN
    own_goal_visible = own_goal_view["count"] >= GOAL_PIXEL_MIN
    opponent_goal_visible = opponent_goal_view["count"] >= GOAL_PIXEL_MIN

    if ball_visible:
        last_ball_x = ball["cx"]
        ball_memory_timer = BALL_MEMORY_STEPS
        search_direction = -1 if ball["cx"] < cam_width / 2.0 else 1
    elif ball_memory_timer > 0:
        ball_memory_timer -= 1

    strong_ball_visible = ball["count"] >= BALL_STRONG_PIXELS
    front_blocked = front_dist < 0.08 or ds_front_left < 560 or ds_front_right < 560
    side_blocked = left_dist < 0.07 or right_dist < 0.07 or ds_left < 650 or ds_right < 650
    back_blocked = ds_back < 650
    near_goal_trap = own_goal_visible and front_dist < 0.18 and not ball_visible
    goal_boxed_in = own_goal_visible and (
        own_goal_view["count"] >= GOAL_ESCAPE_COUNT
        or side_blocked
        or front_dist < 0.11
    )
    wedged = front_dist < 0.05 or (left_dist < 0.055 and right_dist < 0.055)
    own_goal_center_error = abs((own_goal_view["cx"] - cam_width / 2.0) / (cam_width / 2.0))

    if escape_timer == 0 and (wedged or near_goal_trap or (goal_boxed_in and not strong_ball_visible)):
        escape_timer = GOAL_TRAP_STEPS
        if right_dist > left_dist + 0.02:
            escape_direction = 1
        elif left_dist > right_dist + 0.02:
            escape_direction = -1
        else:
            escape_direction = -search_direction

    if escape_timer > 0:
        escape_timer -= 1
        if back_blocked:
            left_speed = ESCAPE_TURN_SPEED if escape_direction > 0 else -ESCAPE_TURN_SPEED
            right_speed = -ESCAPE_TURN_SPEED if escape_direction > 0 else ESCAPE_TURN_SPEED
        else:
            left_speed = -ESCAPE_REVERSE_SPEED
            right_speed = -ESCAPE_REVERSE_SPEED * 0.35
            if escape_direction < 0:
                left_speed, right_speed = right_speed, left_speed
        set_speed(left_speed, right_speed)
        continue

    if ball_visible:
        center_x = cam_width / 2.0
        ball_error = (ball["cx"] - center_x) / center_x
        ball_centered = abs(ball_error) < BALL_CENTER_TOLERANCE
        target_x = ball["cx"]
        own_goal_alignment = abs((ball["cx"] - own_goal_view["cx"]) / cam_width) if own_goal_visible else 1.0
        own_goal_danger = (
            own_goal_visible
            and not opponent_goal_visible
            and own_goal_alignment < OWN_GOAL_DANGER_ALIGNMENT
            and ball["count"] >= BALL_CLOSE_PIXELS
            and (own_goal_center_error < 0.22 or own_goal_view["count"] >= OWN_GOAL_BIG_COUNT)
        )

        if opponent_goal_visible:
            target_x = 0.8 * ball["cx"] + 0.2 * opponent_goal_view["cx"]
        elif own_goal_visible and ball["count"] >= BALL_STRONG_PIXELS:
            mirrored_goal_x = cam_width - own_goal_view["cx"]
            target_x = 0.85 * ball["cx"] + 0.15 * mirrored_goal_x

        drive_speed = CHARGE_SPEED if ball_centered or ball["count"] >= BALL_CLOSE_PIXELS else APPROACH_SPEED
        left_speed, right_speed = steer_to_x(target_x, drive_speed)

        # Once we're lined up on the ball, stop second-guessing and drive through it.
        if own_goal_danger:
            turn_target = cam_width - own_goal_view["cx"]
            left_speed, right_speed = steer_to_x(turn_target, APPROACH_SPEED, turn_gain=4.5)
        elif ball_centered and ball["lowest_y"] > cam_height * 0.45:
            left_speed = CHARGE_SPEED
            right_speed = CHARGE_SPEED
        elif front_blocked and not ball_centered and not strong_ball_visible:
            left_speed, right_speed = steer_to_x(ball["cx"], APPROACH_SPEED * 0.75, turn_gain=7.0)
    elif own_goal_visible and (
        front_blocked or side_blocked or own_goal_view["count"] >= OWN_GOAL_BIG_COUNT
    ):
        turn_target = cam_width - own_goal_view["cx"]
        left_speed, right_speed = steer_to_x(turn_target, SEARCH_TURN_SPEED, turn_gain=6.5)
    elif ball_memory_timer > 0:
        left_speed, right_speed = steer_to_x(last_ball_x, SEARCH_TURN_SPEED, turn_gain=6.0)
    else:
        left_speed = -SEARCH_TURN_SPEED if search_direction > 0 else SEARCH_TURN_SPEED
        right_speed = SEARCH_TURN_SPEED if search_direction > 0 else -SEARCH_TURN_SPEED

    set_speed(left_speed, right_speed)
