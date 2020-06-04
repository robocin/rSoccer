import socket
import struct
from .action_manager import ActionManager

class VSSClient:
    def __init__(self, ip='127.0.0.1', port=54000):
        self.UDP_IP = ip
        self.UDP_PORT = port
        self.frameId = -1
        self.sock = socket.socket(socket.AF_INET, # Internet
                socket.SOCK_DGRAM) # UDP
        self.sock.bind((self.UDP_IP, self.UDP_PORT))
        self.sock.settimeout(2)

        self.action_manager = ActionManager()
        print("Connecting to %s:%d" % (ip, port))
        pass

    # Returns a list of entities
    def unpack(self, data):
        pass

    def receive(self, discardOldFrame=True):

        while True:
            try:
                data, addr = self.sock.recvfrom(1024)  # buffer size is 1024 bytes

                messageType = struct.unpack_from('c', data, 0)[0].decode("utf-8")
                if messageType == 'F':
                    if discardOldFrame:
                        frameId = struct.unpack_from('>i', data, 5)[0]
                        if frameId > self.frameId:
                            self.frameId = frameId
                            break
                    else:
                        break
                elif messageType == 'R':
                    self.action_manager.resume()
                elif messageType == 'P':
                    self.action_manager.pause()
                # elif messageType == 'Y':
                #     self.action_manager.setTeamYellow()
                elif messageType == 'B':
                    self.action_manager.setTeamBlue()
                elif messageType == 'S':
                    self.action_manager.stopRobots()

            except socket.timeout:
                self.frameId = -1

        return data

    def run(self):

        try:
            print("VSSClient :: listening on port {}\n".format(self.UDP_PORT))
            while True:
                self.receive()
        except KeyboardInterrupt:
            print("\nVSSClient :: Client closed.")
        finally:
            pass


def print_about():
    print("This is the VSSClient module.\nPlease use the VSSClient by importing this module into your code.")

if __name__ == '__main__':
    print_about()