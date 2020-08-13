import socket
import gym_ssl.grsim_ssl.pb.messages_robocup_ssl_wrapper_pb2 as wrapper_pb2
import gym_ssl.grsim_ssl.pb.grSim_Packet_pb2 as packet_pb2


class grSimClient:

    def __init__(self, visionIp='224.5.23.2', commandIp='127.0.0.1', visionPort=10020, commandPort=20011):
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
