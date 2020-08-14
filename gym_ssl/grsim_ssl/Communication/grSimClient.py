'''
#  Center all packets communication:
#   - Vision (receives from grSim env) (receives ssl-vision packet + vx vy vw)
    
    bool yellow
    uint32 id
    float kickVx
    float kickVz
    float vx
    float vy
    float vw
    bool dribbler
    bool wheelSpeed
    float vWheel1
    float vWheel2
    float vWheel3
    float vWheel4
'''


import socket
from gym_ssl.grsim_ssl.Entities.Ball import Ball
from gym_ssl.grsim_ssl.Entities.Robot import Robot
import gym_ssl.grsim_ssl.Communication.pb.messages_robocup_ssl_wrapper_pb2 as wrapper_pb2
import gym_ssl.grsim_ssl.Communication.pb.grSim_Packet_pb2 as packet_pb2


class grSimClient:

    def __init__(self, visionIp='224.0.0.1', commandIp='127.0.0.1', visionPort=10020, commandPort=20011):
        """Init grSimClient object."""

        self.visionIp = visionIp
        self.commandIp = commandIp
        self.visionPort = visionPort
        self.commandPort = commandPort

        # Connect vision and command sockets
        self.visionSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.commandSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.visionSocket.bind((self.visionIp, self.visionPort))
        self.commandAddress = (self.commandIp, self.commandPort)

    def send(self, robots):

        packet = packet_pb2.grSim_Packet()
        
        self.__fillPacket(packet, robots)

        """Sends packet to grSim"""
        data = packet.SerializeToString()

        self.commandSocket.sendto(data, self.commandAddress)

    def receive(self):
        """Receive SSL wrapper package and decode."""
        data, _ = self.visionSocket.recvfrom(1024)
        decoded_data = wrapper_pb2.SSL_WrapperPacket().FromString(data)
        
        return decoded_data

    def __fillPacket(self, packet, robots):
        grSimCommands = packet.commands
        grSimCommands.timestamp = 0.0
        grSimRobotCommand = grSimCommands.robot_commands
        for robot in robots:
            rbt = grSimRobotCommand.add()
            rbt.isteamyellow = robot.yellow
            rbt.id = robot.id
            rbt.kickspeedx = robot.kickVx
            rbt.kickspeedz = robot.kickVz
            rbt.veltangent = robot.vx
            rbt.velnormal = robot.vy
            rbt.velangular = robot.vw
            rbt.spinner = robot.dribbler
            rbt.wheelsspeed = robot.wheelSpeed

    
    
    
    
    
    # TEMPORARY TEST
# comm = grSimClient()
# while(True):
#     print(comm.receive())
#     robots = []
#     robots.append(Robot(False, id=0, vx=0, vy=0, vw=2))
#     robots.append(Robot(True, id=0, vx=0, vy=0, vw=2))

#     comm.send(robots)
