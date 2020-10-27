#from envs.gym_soccer.command_pb2 import *
from gym_vss.gym_soccer.state_pb2 import *
from gym_vss.gym_soccer.debug_pb2 import *
import gym_vss.gym_soccer.pb_fira.command_fira_pb2 as command_pb2
import gym_vss.gym_soccer.pb_fira.common_pb2 as common_pb2
import gym_vss.gym_soccer.pb_fira.packet_pb2 as packet_pb2
import gym_vss.gym_soccer.pb_fira.replacement_pb2 as replacement_pb2
import subprocess
from gym_vss.gym_soccer.firaclient import *
import math


'''
agent_init: 0: default positions, 
            1: random positions,
            2: random base positions,
            3: one agent,
            4: goal_keeper,
            5: penalty left,
            6: penalty right
            
ball_init: 0: stopped at the center, 
           1: random slow,
           2: towards left goal,
           3: towards right goal,
           4: towards a random goal
'''


class FiraParser(object):

    def __init__(self, ip=None, port=10002, is_team_yellow=False,
                 team_name='Unknown', manage_process=True,
                 fast_mode=True, render=False):
        # -- Connection
        self.ip = ip
        self.port = port
        self.is_team_yellow = is_team_yellow
        self.team_name = team_name
        self.agent_init = 2
        self.ball_init = 4
        self.fast_mode = fast_mode
        self.render = render
        self.conn = FiraClient(port=self.port)

        # -- Simulator
        self.command_rate = None
        self.simulator_path = None
        self.ia = None
        self.process = None
        self.is_running = False
        self.manage_process = manage_process

    def set_parameters(self, simulator_path, command_rate,
                       ia, agent_init=2, ball_init=4):
        self.simulator_path = simulator_path
        self.command_rate = command_rate
        self.ia = ia
        self.agent_init = agent_init
        self.ball_init = ball_init

    # Simulation methods
    # ----------------------------
    def start(self):
        if self.manage_process:
            self._start_simulation()
        self._connect()
        self.is_running = True
        self.goals_left = 0
        self.goals_right = 0

    def stop(self):
        self.is_running = False
        self._disconnect()
        if self.manage_process:
            self._stop_simulation()

    def reset(self):
        self.stop()
        self.start()

    def _start_simulation(self):
        command = [self.simulator_path]
        if not self.render:
            command.append('-H')
        if self.fast_mode:
            command.append('--xlr8')
        self.process = subprocess.Popen(command)

    def _stop_simulation(self):
        self.process.terminate()
        self.process.wait()

    # Network methods
    # ----------------------------
    def convert_ssl_to_sim_coord(self, pose, v_pose):
        width = 1.3/2.0
        lenght = (1.5/2.0) + 0.1
        if(self.is_team_yellow):
            pose.x = (lenght - pose.x)*100
            pose.y = (width + pose.y)*100
            pose.yaw = math.pi - pose.yaw
            v_pose.x *= -100
            v_pose.y *= 100
            v_pose.yaw = v_pose.yaw
        else:
            pose.x = (lenght+pose.x)*100
            pose.y = (width - pose.y)*100
            pose.yaw = -pose.yaw
            v_pose.x *= 100
            v_pose.y *= -100
            v_pose.yaw = v_pose.yaw

    def receive(self):
        data = self.conn.receive()
        state = Global_State()
        state.time = data.step
        state.goals_yellow = data.goals_yellow
        state.goals_blue = data.goals_blue

        # Ball
        ball = data.frame.ball
        ball_state = state.balls.add()
        ball_state.pose.x = ball.x
        ball_state.pose.y = ball.y
        ball_state.v_pose.x = ball.vx
        ball_state.v_pose.y = ball.vy
        self.convert_ssl_to_sim_coord(ball_state.pose, ball_state.v_pose)

        # Robots Yellow
        for robot in data.frame.robots_yellow:
            robot_state = state.robots_yellow.add()
            robot_state.pose.x = robot.x
            robot_state.pose.y = robot.y
            robot_state.pose.yaw = robot.orientation
            robot_state.v_pose.x = robot.vx
            robot_state.v_pose.y = robot.vy
            robot_state.v_pose.yaw = robot.vorientation
            self.convert_ssl_to_sim_coord(robot_state.pose,
                                          robot_state.v_pose)

        # Robots Blue
        for robot in data.frame.robots_blue:
            robot_state = state.robots_blue.add()
            robot_state.pose.x = robot.x
            robot_state.pose.y = robot.y
            robot_state.pose.yaw = robot.orientation
            robot_state.v_pose.x = robot.vx
            robot_state.v_pose.y = robot.vy
            robot_state.v_pose.yaw = robot.vorientation
            self.convert_ssl_to_sim_coord(robot_state.pose,
                                          robot_state.v_pose)
        # print(state.robots_blue[0].v_pose)

        return state

    def send_speeds(self, team):
        # prepare commands
        pkt = packet_pb2.Packet()
        d = pkt.cmd.robot_commands

        # send wheel speed commands for each robot
        for i in range(0, len(team)):
            robot = d.add()
            robot.id = i
            robot.yellowteam = self.is_team_yellow
            robot.wheel_left = team[i].left_wheel_speed
            robot.wheel_right = team[i].right_wheel_speed

        # send commands
        data = pkt.SerializeToString()
        self.com_socket.sendto(data, self.address)

    def send_debug(self, team):
        d = Global_Debug()

        for i in range(0, len(team)):
            pose = d.final_poses.add()
            pose.id = i
            pose.x = team[i].target_x
            pose.y = team[i].target_y
            pose.yaw = team[i].target_theta

        data = d.SerializeToString()
        self.socket_debug.send(data)

    def _disconnect(self):
        pass

    def _connect(self):
        self.com_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.address = ("127.0.0.1", self.port+1)
        self.conn.connect()
