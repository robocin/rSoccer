import socket
from typing import Dict, List

import numpy as np
from rsoccer_gym.Entities import Robot, Field
from rsoccer_gym.Entities.Frame import FramePB
from rsoccer_gym.Simulators.rsim import RSim

import rsoccer_gym.Simulators.pb_fira.packet_pb2 as packet_pb2
from rsoccer_gym.Simulators.pb_fira.state_pb2 import *


class Fira(RSim):
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
        self.vision_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 128)
        self.vision_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        self.vision_sock.bind((self.vision_ip, self.vision_port))
        self.vision_sock.setblocking(False)
        self.linear_speed_range = 1.15
        self.robot_wheel_radius = 0.026

    def get_field_params(self):
        # Using VSSSLeague/FIRASim tag v3.0 default values
        # Max rpm was adjusted to match the rsim max v
        return Field(
            length=1.5,
            width=1.3,
            penalty_length=0.15,
            penalty_width=0.7,
            goal_width=0.4,
            goal_depth=0.1,
            ball_radius=0.0215,
            rbt_distance_center_kicker=-1.0,
            rbt_kicker_thickness=-1.0,
            rbt_kicker_width=-1.0,
            rbt_wheel0_angle=90.0,
            rbt_wheel1_angle=270.0,
            rbt_wheel2_angle=-1.0,
            rbt_wheel3_angle=-1.0,
            rbt_radius=0.0375,
            rbt_wheel_radius=0.02,
            rbt_motor_max_rpm=572.0,
        )

    def stop(self):
        pass

    def reset(self, frame: FramePB):
        pkt = packet_pb2.Packet()

        # replacement commands to stop robots
        cmd_pkt = pkt.cmd.robot_commands
        robots_pkt = pkt.replace.robots
        ball_pkt = pkt.replace.ball

        ball_pkt.x = frame.ball.x
        ball_pkt.y = frame.ball.y

        for id, robot in frame.robots_blue.items():
            rep_rob = robots_pkt.add()
            rep_rob.position.robot_id = id
            rep_rob.position.x = robot.x
            rep_rob.position.y = robot.y
            rep_rob.position.orientation = robot.theta
            rep_rob.yellowteam = False
            rep_rob.turnon = True

            cmd_rob = cmd_pkt.add()
            cmd_rob.id = id
            cmd_rob.yellowteam = False
            cmd_rob.wheel_left = 0.0
            cmd_rob.wheel_right = 0.0

        for id, robot in frame.robots_yellow.items():
            rep_rob = robots_pkt.add()
            rep_rob.position.robot_id = id
            rep_rob.position.x = robot.x
            rep_rob.position.y = robot.y
            rep_rob.position.orientation = robot.theta
            rep_rob.yellowteam = True
            rep_rob.turnon = True

            cmd_rob = cmd_pkt.add()
            cmd_rob.id = id
            cmd_rob.yellowteam = True
            cmd_rob.wheel_left = 0.0
            cmd_rob.wheel_right = 0.0

        # send commands
        data = pkt.SerializeToString()
        self.com_socket.sendto(data, self.com_address)

    def get_frame(self):
        """Receive package and decode."""
        data = None
        while True:
            try:
                data, _ = self.vision_sock.recvfrom(1024)
            except:
                if data != None:
                    break
        decoded_data = packet_pb2.Environment().FromString(data)
        frame = FramePB()
        frame.parse(decoded_data)
        return frame

    def send_commands(self, commands):
        # prepare commands
        pkt = packet_pb2.Packet()
        d = pkt.cmd.robot_commands

        # send wheel speed commands for each robot
        for cmd in commands:
            robot = d.add()
            robot.id = cmd.id
            robot.yellowteam = cmd.yellow

            # convert from linear speed to angular speed
            robot.wheel_left = cmd.v_wheel0
            robot.wheel_right = cmd.v_wheel1

        # send commands
        data = pkt.SerializeToString()
        self.com_socket.sendto(data, self.com_address)
    
