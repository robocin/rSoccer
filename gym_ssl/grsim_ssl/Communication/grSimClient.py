'''
#  Center all packets communication:
#   - Vision (receives from grSim env) (receives ssl-vision packet + vx vy vw)

'''


import socket
import gym_ssl.grsim_ssl.Communication.pb.messages_robocup_ssl_wrapper_pb2 as wrapper_pb2
import gym_ssl.grsim_ssl.Communication.pb.grSim_Packet_pb2 as packet_pb2
from gym_ssl.grsim_ssl.Entities import Robot, Ball, Frame

class grSimClient:

    def __init__(self, visionIp='224.0.0.1', commandIp='127.0.0.1', visionPort=10020, commandPort=20011):
        """Init grSimClient object."""

        self.visionIp = visionIp
        self.commandIp = commandIp
        self.visionPort = visionPort
        self.commandPort = commandPort

        # Connect vision and command sockets
        self.visionSocket = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.visionSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.visionSocket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 128)
        self.visionSocket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)

        self.commandSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.visionSocket.bind((self.visionIp, self.visionPort))
        self.commandAddress = (self.commandIp, self.commandPort)

        

#-------------------------------------------------------------------------    

    def receiveState(self):
        """Receive SSL wrapper package and decode."""
        data, _ = self.visionSocket.recvfrom(1024)
        decoded_data = wrapper_pb2.SSL_WrapperPacket().FromString(data)
        
        frame = Frame()
        frame.parse(decoded_data)

        return frame

#-------------------------------------------------------------------------    
    

    def sendCommandsPacket(self, commands):
        packet = self._fillCommandPacket(commands)
        
        """Sends packet to grSim"""
        data = packet.SerializeToString()

        self.commandSocket.sendto(data, self.commandAddress)


    def _fillCommandPacket(self, commands):
        packet = packet_pb2.grSim_Packet()
        grSimCommands = packet.commands
        grSimCommands.timestamp = 0.0
        grSimRobotCommand = grSimCommands.robot_commands
        for robotCommand in commands:
            rbt = grSimRobotCommand.add()
            rbt.isteamyellow = robotCommand.yellow
            rbt.id = robotCommand.id
            rbt.kickspeedx = robotCommand.kickVx
            rbt.kickspeedz = robotCommand.kickVz
            rbt.veltangent = robotCommand.vx
            rbt.velnormal = robotCommand.vy
            rbt.velangular = robotCommand.vw
            rbt.spinner = robotCommand.dribbler
            rbt.wheelsspeed = robotCommand.wheelSpeed
        return packet

#-------------------------------------------------------------------------    

    def sendReplacementPacket(self, robotPositions = None, ballPosition = None):
        packet = self._fillReplacementPacket(robotPositions, ballPosition)
        """Sends packet to grSim"""
        data = packet.SerializeToString()

        self.commandSocket.sendto(data, self.commandAddress)


    def _fillReplacementPacket(self, robotPositions = None, ballPosition = None):
        packet = packet_pb2.grSim_Packet()
        grSimReplacement = packet.replacement
        
        if ballPosition != None:
            grSimBall = grSimReplacement.ball
            grSimBall.x = ballPosition.x
            grSimBall.y = ballPosition.y
            grSimBall.vx = ballPosition.vx
            grSimBall.vy = ballPosition.vy
        
        if robotPositions != None:
            grSimRobot = grSimReplacement.robots
            for rbtPosition in robotPositions:
                rbt = grSimRobot.add()
                rbt.yellowteam = rbtPosition.yellow
                rbt.id = rbtPosition.id
                rbt.x = rbtPosition.x
                rbt.y = rbtPosition.y
                rbt.dir = rbtPosition.theta
        return packet