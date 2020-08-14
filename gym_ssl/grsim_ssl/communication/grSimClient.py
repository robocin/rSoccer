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
    bool wheelsspeed
    float vWheel1
    float vWheel2
    float vWheel3
    float vWheel4


'''


import socket
import gym_ssl.grsim_ssl.communication.pb.messages_robocup_ssl_wrapper_pb2 as wrapper_pb2
import gym_ssl.grsim_ssl.communication.pb.grSim_Packet_pb2 as packet_pb2


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

    def send(self, packet):
        """Sends packet to grSim"""
        data = packet.SerializeToString()
        self.commandSocket.sendto(data, self.commandAddress)

    def receive(self):
        """Receive SSL wrapper package and decode."""
        data, _ = self.visionSocket.recvfrom(1024)
        decoded_data = wrapper_pb2.SSL_WrapperPacket().FromString(data)
        
        return decoded_data

    def encode_packet(self, actions):
        return "NOT IMPLEMENTED"
    
    
    
    
    
    # TEMPORARY TEST
    # comm = grSimClient()
    # while(True):
    # print(comm.receive())
    
    # packet = packet_pb2.grSim_Packet()
    # grSimCommands = packet.commands
    # grSimRobotCommand = grSimCommands.robot_commands
    # grSimCommands.timestamp = 0.0
    # robot = grSimRobotCommand.add()
    # robot.isteamyellow = False
    # robot.id = 0
    # robot.kickspeedx = 0
    # robot.kickspeedz = 0
    # robot.veltangent = 0
    # robot.velnormal = 0
    # robot.velangular = 2
    # robot.spinner = False
    # robot.wheelsspeed = False

    # robot = grSimRobotCommand.add()
    # robot.isteamyellow = True
    # robot.id = 0
    # robot.kickspeedx = 0
    # robot.kickspeedz = 0
    # robot.veltangent = 0
    # robot.velnormal = 0
    # robot.velangular = 2
    # robot.spinner = False
    # robot.wheelsspeed = False

    # comm.send(packet)
