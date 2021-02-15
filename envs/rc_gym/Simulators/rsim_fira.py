import pb_fira.packet_pb2 as packet_pb2
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import collections
import argparse
import socket
import torch
import ptan
import time
import gym
import os

from rc_gym.Entities.FramePB import FramePB
from rc_gym.Entities import Robot
from pb_fira.state_pb2 import *
from rsim_base import RSim


class RSimFira(RSim):
    def __init__(
        self,
        vision_ip="224.0.0.1",
        vision_port=10002,
        cmd_ip="127.0.0.1",
        cmd_port=20011,
    ):
        """
        Init SSLClient object.
        Extended description of function.
        Parameters
        ----------
        ip : str
            Multicast IP in format '255.255.255.255'. 
        port : int
            Port up to 1024. 
        """

        self.vision_ip = vision_ip
        self.vision_port = vision_port
        self.com_ip = cmd_ip
        self.com_port = cmd_port
        self.com_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.com_address = (self.com_ip, self.com_port)

        self.vision_sock = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
        )
        self.vision_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.vision_sock.setsockopt(
            socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 128
        )
        self.vision_sock.setsockopt(
            socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1
        )
        self.vision_sock.bind((self.vision_ip, self.vision_port))

    def receive(self, id):
        """Receive package and decode."""

        data, _ = self.vision_sock.recvfrom(1024)
        decoded_data = packet_pb2.Environment().FromString(data)
        self.frame = FramePB()
        self.frame.parse(decoded_data)

        return self._frame_to_observations(id)

    def send(self, commands):
        # prepare commands
        pkt = packet_pb2.Packet()
        d = pkt.cmd.robot_commands

        # send wheel speed commands for each robot
        for cmd in commands:
            robot = d.add()
            robot.id = cmd.id
            robot.yellowteam = cmd.yellow

            # convert from linear speed to angular speed
            robot.wheel_left = cmd.v_wheel1
            robot.wheel_right = cmd.v_wheel2

        # send commands
        data = pkt.SerializeToString()
        self.com_socket.sendto(data, self.com_address)

    def send_commands(self, commands):
        sim_commands = np.zeros(
            (self.n_robots_blue + self.n_robots_yellow, 2), dtype=np.float64
        )

        for cmd in commands:
            if cmd.yellow:
                rbt_id = self.n_robots_blue + cmd.id
            else:
                rbt_id = cmd.id

            # Convert from linear speed to angular speed
            sim_commands[rbt_id][0] = cmd.v_wheel1 / self.robot_wheel_radius
            sim_commands[rbt_id][1] = cmd.v_wheel2 / self.robot_wheel_radius

        self.simulator.step(sim_commands)

    def get_frame(self) -> FramePB:
        state = self.simulator.get_state()
        # Update frame with new state
        frame = FramePB()
        frame.parse(state, self.n_robots_blue, self.n_robots_yellow)

        return frame


ENV = "VSS3v3-v0"
DEADZONE = 0.05
SPEED_RANGE = 1.15
WHEEL_RADIUS = 0.026
# Q_SIZE = 1


class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def _actions_to_v_wheels(actions):
    left_wheel_speed = actions[0] * SPEED_RANGE
    right_wheel_speed = actions[1] * SPEED_RANGE

    # Deadzone
    if -DEADZONE < left_wheel_speed < DEADZONE:
        left_wheel_speed = 0

    if -DEADZONE < right_wheel_speed < DEADZONE:
        right_wheel_speed = 0

    left_wheel_speed, right_wheel_speed = np.clip(
        (left_wheel_speed, right_wheel_speed), -2.6, 2.6
    )

    return left_wheel_speed / WHEEL_RADIUS, right_wheel_speed / WHEEL_RADIUS


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", required=True, help="Model file to load"
    )
    args = parser.parse_args()

    net = DDPGActor(40, 2)
    net.load_state_dict(torch.load(args.model))

    # client = FiraClient()
    # client = RSim()
    client = RSimFira()

    while True:
        obs_v = torch.FloatTensor(
            [client.receive(0), client.receive(1), client.receive(2)]
        )

        print(client.receive(0).shape())

        mu_v = net(obs_v)
        action = mu_v.squeeze(dim=0).data.numpy()
        action = np.clip(action, -1, 1)
        command = []
        for i in range(3):
            lw, rw = _actions_to_v_wheels(action[i])
            command.append(Robot(yellow=False, id=i, v_wheel1=lw, v_wheel2=rw))
        client.send(command)
