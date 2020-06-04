import sslclient

from envs.gym_soccer.state_pb2 import *
from envs.speed_estimator import *


def convert_ssl_to_sim_coord(pose):
    pose.x = pose.x / 10.0 + 75 + 10
    pose.y = pose.y / 10.0 + 65
    pose.y = 130 - pose.y
    pose.yaw = -pose.yaw


class SSLParser:
    def __init__(self, ip, port):
        self.conn = sslclient.client(ip, port)
        self.conn.connect()
        self.reset()

    def reset(self):
        self.goal_left = 0
        self.goal_right = 0
        self.prev_ball = None

    def receive(self):
        data = self.conn.receive()
        return self.ssl_to_state_pkt(data)

    def ssl_to_state_pkt(self, pkt):
        state = Global_State()
        state.time = int(pkt.detection.t_capture*1000)

        # print("balls: ")
        # print(pkt.detection.balls)

        for ball in pkt.detection.balls:
            ball_state = state.balls.add()
            ball_state.pose.x = ball.x
            ball_state.pose.y = ball.y
            convert_ssl_to_sim_coord(ball_state.pose)

        if len(state.balls) == 0:  # lost ball (goal?)
            if self.prev_ball is not None and 45 < self.prev_ball.pose.y < 85:
                if self.prev_ball.pose.x > 156:  # yes, left goal
                    self.goal_left += 1
                    self.prev_ball = None
                    print("GOAL LEFT! score: %d x %d" % (self.goal_left, self.goal_right))
                elif self.prev_ball.pose.x < 14:  # yes, right goal
                    self.goal_right += 1
                    self.prev_ball = None
                    print("GOAL RIGHT! score: %d x %d" % (self.goal_left, self.goal_right))

            ball_state = state.balls.add()
            ball_state.pose.x = 85.0
            ball_state.pose.y = 65.0
        elif self.prev_ball is not None or 22 < ball_state.pose.x < 146:
            self.prev_ball = ball_state  # only enable a new goal if the ball is seen far from goal again

        state.goals_blue = self.goal_left
        state.goals_yellow = self.goal_right

        #print("ball:", ball_state, self.prev_ball is None, "goals: ", self.goal_left, " x ", self.goal_right)


            #print(" --------- LOST BALL ----------")
        # print("robots yellow: ")
        # print(pkt.detection.robots_yellow)

        for idx, robot in enumerate(sorted(pkt.detection.robots_yellow, key = lambda robot: robot.robot_id)):
            robot_state = state.robots_yellow.add()
            robot_state.pose.x = robot.x
            robot_state.pose.y = robot.y
            robot_state.pose.yaw = robot.orientation
            convert_ssl_to_sim_coord(robot_state.pose)

        while len(state.robots_yellow) < 3:
            #print(" --------- LOST YELLOW ROBOT ----------")
            robot_state = state.robots_yellow.add()
            robot_state.pose.x = 5
            robot_state.pose.y = 65

        # print("robots blue: ")
        # print(pkt.detection.robots_blue)
        for idx, robot in enumerate(sorted(pkt.detection.robots_blue, key = lambda robot: robot.robot_id)):

            robot_state = state.robots_blue.add()
            robot_state.pose.x = robot.x
            robot_state.pose.y = robot.y
            robot_state.pose.yaw = robot.orientation
            convert_ssl_to_sim_coord(robot_state.pose)

        while len(state.robots_blue) < 3:
            #print(" --------- LOST BLUE ROBOT ----------")
            robot_state = state.robots_blue.add()
            robot_state.pose.x = 5
            robot_state.pose.y = 65

        return state

