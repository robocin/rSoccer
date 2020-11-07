from rc_gym.vss.Simulators.vss_sdk.pb2.command_pb2 import *
from rc_gym.vss.Simulators.vss_sdk.pb2.state_pb2 import *
from rc_gym.vss.Simulators.vss_sdk.pb2.debug_pb2 import *
from rc_gym.Entities import FrameSDK

import subprocess
import os
import zmq

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


class SimulatorVSS:
    def __init__(self, field_type: int,
                 n_robots_blue: int, n_robots_yellow: int):
        self.port = 5555
        self._connect()
        self._start_simulation()

    def reset(self):
        self._disconnect()
        self.simulator.terminate()
        self.simulator.wait()
        self._start_simulation()
        self._connect()

    def send_commands(self, commands):
        # prepare commands
        command_pb = Global_Commands()
        command_pb.id = 0
        command_pb.is_team_yellow = False
        command_pb.situation = 0
        command_pb.name = 'RoboCin Deep'

        # send wheel speed commands for each robot
        for cmd in commands:
            if not cmd.yellow:
                robot = command_pb.robot_commands.add()
                robot.id = cmd.i
                robot.left_vel = cmd.v_wheel1 * 100
                robot.right_vel = cmd.v_wheel2 * 100

        # send commands
        data = command_pb.SerializeToString()
        self.socket_com.send(data)

    def replace_from_frame(self, frame):
        debug_pb = Global_Debug()

        for i, robot in enumerate(frame.robots_blue.values()):
            pose = debug_pb.final_poses.add()
            pose.id = i
            pose.x = robot.x
            pose.y = robot.y
            pose.yaw = robot.theta

        data = debug_pb.SerializeToString()
        self.socket_debug.send(data)

    def get_frame(self):
        state = Global_State()
        data = self.state_socket.recv()
        count = 0
        while count < 100:
            socks = dict(self.poller.poll(1))
            if self.state_socket in socks and socks[self.state_socket] == zmq.POLLIN:
                # discard messages
                data = self.state_socket.recv()
                count += 1
            else:
                break

        state.ParseFromString(data)

        frame = FrameSDK()
        frame.parse(state)
        
        return frame
    
    def get_field_params(self):
        return {'field_width': 1.3, 'field_length': 1.5, 'penalty_width': 0.7, 'penalty_length': 0.15, 'goal_width': 0.4, 'goal_depth': 0.1}

    def _start_simulation(self):
        path = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                            'bin', 'VSS-SimulatorSync')
        ia = '0'
        command_rate = '1'
        agent_init = '1'
        self.simulator = subprocess.Popen(
            [path, '-d', '-i', ia, '-r', command_rate, '-p', str(self.port),
             '-a', agent_init])

    def _disconnect(self):
        if self.poller is not None:
            self.poller.unregister(self.state_socket)
            self.state_socket.close()
            self.socket_com.close()
            self.socket_debug.close()
            self.context.term()
            self.poller = None

    def _connect(self):
        self.context = zmq.Context()
        self.state_socket = self.context.socket(zmq.SUB)
        self.state_socket.setsockopt(zmq.CONFLATE, 1)
        self.state_socket.connect("tcp://127.0.0.1:%d" % self.port)

        try:
            self.state_socket.setsockopt(zmq.SUBSCRIBE, b'')
        except TypeError:
            self.state_socket.setsockopt_string(zmq.SUBSCRIBE, b'')
        self.state_socket.setsockopt(zmq.LINGER, 0)
        self.state_socket.setsockopt(zmq.RCVTIMEO, 5000)

        # commands socket
        self.socket_com = self.context.socket(zmq.PAIR)  # socket to Team
        # if self.is_team_yellow:
        #     self.socket_com.connect("tcp://127.0.0.1:%d" % (self.port + 1))
        # else:
        self.socket_com.connect("tcp://127.0.0.1:%d" % (self.port + 2))

        # debugs socket
        self.socket_debug = self.context.socket(
            zmq.PAIR)  # debug socket to Team 1
        # if self.is_team_yellow:
        #     self.socket_debug.connect("tcp://127.0.0.1:%d" % (self.port + 3))
        # else:
        self.socket_debug.connect("tcp://127.0.0.1:%d" % (self.port + 4))

        # Initialize poll set
        self.poller = zmq.Poller()
        self.poller.register(self.state_socket, zmq.POLLIN)
