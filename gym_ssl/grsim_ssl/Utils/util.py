import math
import numpy as np

# Constants
# ----------------------------

CTRL_DEFAULT = 0
CTRL_COLLECT_LOG = 2
CTRL_RANDOM_WALK = 4
CTRL_CORRECT = 8
CTRL_SPLINE = 16
CTRL_RANDBALL = 32
CTRL_CUSTOM_SPEEDS = 64

#ctrl_mode = CTRL_COLLECT_LOG | CTRL_CORRECT
#ctrl_mode = CTRL_COLLECT_LOG | CTRL_DEFAULT
#ctrl_mode = CTRL_COLLECT_LOG | CTRL_SPLINE
#ctrl_mode = CTRL_COLLECT_LOG | CTRL_RANDBALL
#ctrl_mode = CTRL_COLLECT_LOG | CTRL_CUSTOM_SPEEDS


# ctrl_mode = CTRL_CORRECT
# ctrl_mode = CTRL_SPLINE
ctrl_mode = CTRL_DEFAULT

dict_ctrl_model_id = {0:"ctrl/models/ctrl_log_custom_0_2019-10-09_16_47_19_processed.cpth",1:"ctrl/models/custom_ctrl_log0_2019-09-12_09_08_42_processed.cpth",2:"ctrl/models/custom_ctrl_log0_2019-09-12_09_08_42_processed.cpth"}
ctrl_model_input_size = 4
ctrl_model_output_size = 2

field = None
fig = None
ax = None


# Math methods
# ----------------------------

def mod(x, y):
    return (math.sqrt(x*x + y*y))


def angle(x, y):
    return math.atan2(y, x)


def to_positive_angle(angle):
    return math.fmod(angle + 2 * math.pi, 2 * math.pi)


def smallest_angle_diff(target, source):
    a = to_positive_angle(target) - to_positive_angle(source)

    if a > math.pi:
        a = a - 2 * math.pi
    elif a < -math.pi:
        a = a + 2 * math.pi

    return a

def to_pi_range(angle):
    angle = math.fmod(angle, 2 * math.pi)
    if angle < -math.pi:
        angle = angle + 2 * math.pi
    elif angle > math.pi:
        angle = angle - 2 * math.pi

    return angle


def clip(val, vmin, vmax):
    return min(max(val, vmin), vmax)


def normX(x):
    return clip(x / 170.0, -0.2, 1.2)


def normVx(vx):
    return clip(vx / 80.0, -1.25, 1.25)


def normVt(vt):
    return clip(vt / 10, -1.2, 1.2)


def roundTo5(x, base=5):
    return int(base * round(float(x) / base))


# Env aux methods
# ----------------------------


def transpose_state(state):
    max_x = 170
    max_y = 130

    for _, ball in enumerate(state.balls):
        ball.pose.x = max_x - ball.pose.x
        ball.pose.y = max_y - ball.pose.y
        ball.v_pose.x = -ball.v_pose.x
        ball.v_pose.y = -ball.v_pose.y

    for _, robot_yellow in enumerate(state.robots_yellow):
        robot_yellow.pose.x = max_x - robot_yellow.pose.x
        robot_yellow.pose.y = max_y - robot_yellow.pose.y
        robot_yellow.pose.yaw = to_pi_range(robot_yellow.pose.yaw + math.pi)

        robot_yellow.v_pose.x = -robot_yellow.v_pose.x
        robot_yellow.v_pose.y = -robot_yellow.v_pose.y

    for _, robot_blue in enumerate(state.robots_blue):
        robot_blue.pose.x = max_x - robot_blue.pose.x
        robot_blue.pose.y = max_y - robot_blue.pose.y
        robot_blue.pose.yaw = to_pi_range(robot_blue.pose.yaw + math.pi)

        robot_blue.v_pose.x = -robot_blue.v_pose.x
        robot_blue.v_pose.y = -robot_blue.v_pose.y
        
    return state


# Debug methods
# ----------------------------


def getTransformMatrix_l2g(theta, pos_x, pos_y):#global 2 local transform
    R_l2g = np.array([[math.cos(theta),math.sin(theta)],
                    [-math.sin(theta),math.cos(theta)]]) #2x2
    R_g2l = np.transpose(R_l2g)

    D = np.array([pos_x,pos_y]) #2x1

    H_l2g = np.column_stack((R_l2g, D))  # 2x3
    H_l2g = np.row_stack((H_l2g, np.array([0,0,1]))) #3x3

    return H_l2g


#consider 8cm size and 1cm = 1 pixel
def get_robots_bounds(robot, scale):
    #matrix transform
    H = getTransformMatrix_l2g(robot["theta"], robot["x"]*scale, robot["y"]*scale)

    dist = 4*scale
    left_up_corner = np.matmul(H, np.array((-dist, -dist, 1)))[0:2].tolist()
    left_down_corner = np.matmul(H, np.array((-dist, dist, 1)))[0:2].tolist()
    right_up_corner = np.matmul(H, np.array((dist, -dist, 1)))[0:2].tolist()
    right_down_corner = np.matmul(H, np.array((dist, dist, 1)))[0:2].tolist()

    return np.array([left_up_corner, left_down_corner, right_down_corner, right_up_corner])


# def draw_robots(field_img, bound_box, is_my_team):
#     center1 = np.array([sum(y) / len(y) for y in zip(*(bound_box[0], bound_box[1]))], dtype=np.int32)
#     center2 = np.array([sum(y) / len(y) for y in zip(*(bound_box[2], bound_box[3]))], dtype=np.int32)
#     center3 = np.array([sum(y) / len(y) for y in zip(*(bound_box[0], bound_box[3]))], dtype=np.int32)
#     center4 = np.array([sum(y) / len(y) for y in zip(*(bound_box[1], bound_box[2]))], dtype=np.int32)
    
#     if(not is_my_team):
#         cv2.polylines(field_img, np.int32([bound_box]), True, [255,255,255])
#         points = np.array([bound_box[2],bound_box[3],center3,center4], dtype=np.int32)
#         cv2.polylines(field_img, [points],True, [255,255,255])

#     else:
#         cv2.polylines(field_img, np.int32([bound_box]), True, [255,255,255])
#         points = np.array([center3, center2, center4], dtype=np.int32)
#         cv2.polylines(field_img, [points], True, [255,255,255])


# def cntrl_debug(robot, target):
     
#     scale = 3
    
#     cv2.namedWindow('target', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('target', scale*172, scale*132)
#     field_img = np.zeros([130*scale, 170*scale], np.uint8)
    
#     field_dimensions = np.array([[10*scale,1*scale],[160*scale,1*scale],[160*scale,50*scale],[169*scale,50*scale],
#                         [169*scale,80*scale],[160*scale,80*scale],[160*scale,129*scale],[10*scale,129*scale],
#                         [10*scale,80*scale],[1*scale,80*scale],[1*scale,50*scale],[10*scale,50*scale]],np.int32)
#     robots_positions = [robot]

#     cv2.polylines(field_img, [field_dimensions], True, [255, 255, 255])
#     bound_box = get_robots_bounds(robots_positions[0], scale)
#     draw_robots(field_img, bound_box, True)
#     ball = {"x":target["x"], "y":target["y"], "r":1}

#     cv2.circle(field_img,(np.int32(ball["x"])*scale,np.int32(ball["y"])*scale), ball["r"]*scale, (255,255,255), -1)

#     cv2.imshow('target', field_img)
#     cv2.waitKey(1)
