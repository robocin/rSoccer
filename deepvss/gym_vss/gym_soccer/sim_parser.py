from gym_vss.gym_soccer.command_pb2 import *
from gym_vss.gym_soccer.state_pb2 import *
from gym_vss.gym_soccer.debug_pb2 import *
import subprocess
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
class SimParser(object):

    def __init__(self, ip=None, port=None, is_team_yellow=False, team_name='Unknown', manage_process=True):
        # -- Connection
        self.ip = ip
        self.port = port
        self.is_team_yellow = is_team_yellow
        self.team_name = team_name
        self.agent_init = 2
        self.ball_init = 4

        # -- Simulator
        self.command_rate = None
        self.simulator_path = None
        self.ia = None
        self.process = None
        self.is_running = False
        self.manage_process = manage_process

    def set_parameters(self, simulator_path, command_rate, ia, agent_init=2, ball_init=4):
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

    def stop(self):
        self.is_running = False
        self._disconnect()
        if self.manage_process:
            self._stop_simulation()

    def reset(self):
        self.stop()
        self.start()

    def _start_simulation(self):
        self.process = subprocess.Popen(
            [self.simulator_path, '-d', '-i',
             str(int(self.ia)), '-r', str(self.command_rate), '-p', str(self.port), '-a', str(self.agent_init)])

    def _stop_simulation(self):
        self.process.terminate()
        self.process.wait()

    # Network methods
    # ----------------------------

    def receive(self):
        try:
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

            return state

        except Exception as e:
            print("Caught timeout:" + str(e))
            return None

    def send_speeds(self, team):
        # prepare commands
        c = Global_Commands()
        c.id = 0
        c.is_team_yellow = self.is_team_yellow
        c.situation = 0
        c.name = self.team_name

        # send wheel speed commands for each robot
        for i in range(0, len(team)):
            robot = c.robot_commands.add()
            robot.id = i
            robot.left_vel = team[i].left_wheel_speed
            robot.right_vel = team[i].right_wheel_speed

        # send commands
        data = c.SerializeToString()
        self.socket_com.send(data)

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
        if self.is_team_yellow:
            self.socket_com.connect("tcp://127.0.0.1:%d" % (self.port + 1))
        else:
            self.socket_com.connect("tcp://127.0.0.1:%d" % (self.port + 2))

        # debugs socket
        self.socket_debug = self.context.socket(zmq.PAIR)  # debug socket to Team 1
        if self.is_team_yellow:
            self.socket_debug.connect("tcp://127.0.0.1:%d" % (self.port + 3))
        else:
            self.socket_debug.connect("tcp://127.0.0.1:%d" % (self.port + 4))

        # Initialize poll set
        self.poller = zmq.Poller()
        self.poller.register(self.state_socket, zmq.POLLIN)

