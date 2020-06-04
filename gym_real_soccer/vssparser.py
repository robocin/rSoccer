import time
import struct
from envs.gym_real_soccer.vssclient import VSSClient

from envs.gym_soccer.state_pb2 import *
from envs.speed_estimator import *
import math
from envs.utils import to_pi_range
from .proto_state_models import *
from .action_manager import ActionManager

class VSSParser:
    def __init__(self, ip, port):
        self.conn = VSSClient(ip, port)
        self.start_time = time.time()
        self.header_offset = 10
        self.entity_size = 8 * 3 + 1
        self.reset()

        self.goal_left = 0
        self.goal_right = 0
        self.prev_ball = None

        self.yellow_id = dict()
        self.blue_id = dict()

        self.state = ProtoState()
        self.state.balls = [EntityState()]
        self.state.balls[0].pose.x = 85
        self.state.balls[0].pose.y = 65

        self.state.robots_yellow = [EntityState() for _ in range(3)]
        self.state.robots_blue = [EntityState() for _ in range(3)]
        for i in range(0, 3):
            self.state.robots_yellow[i].pose.x = 5
            self.state.robots_yellow[i].pose.y = 65

        for i in range(0, 3):
            self.state.robots_blue[i].pose.x = 5
            self.state.robots_blue[i].pose.y = 65

        self.action_manager = ActionManager()

    def reset(self):
        self.goal_left = 0
        self.goal_right = 0
        self.prev_ball = None

    def receive(self):
        data = self.conn.receive()
        return self.vss_to_state_pkt(data, debug=False)

    def vss_to_state_pkt(self, data, debug=False):

        self.state.time = struct.unpack_from('>I', data, 1)[0]
        #frameId = struct.unpack_from('>i', data, 5)[0]
        entities = ord(struct.unpack_from('c', data, self.header_offset-1)[0])

        # if debug: print("message have {} entities on frame {}".format(entities, frameId))

        found_ball = False
        op_id = 0
        for i in range(0, entities):  # Parse all entities
            id = ord(struct.unpack_from('c', data, self.header_offset + self.entity_size * i)[0])
            x, y, ang = struct.unpack_from('ddd', data, self.header_offset + 1 + self.entity_size * i)
            # if debug:
            #     print("id {} position ({},{},{})".format(id, x, y, ang))

            # print("id:", id, x, y, -ang)
            if id == 0:  # BALLS
                if x > 8 and x < 162:
                    self.state.balls[0].pose.x = x
                    self.state.balls[0].pose.y = 130-y
                    found_ball = True
                    # if debug:
                    #print("ball:", self.state.balls[0].pose.x, self.state.balls[0].pose.y)
                else:
                    self.state.balls[0].pose.x = 85
                    self.state.balls[0].pose.y = 65

            elif id // 200 == 1:
                # Yellow Team
                    if id not in self.yellow_id:
                        n = len(self.yellow_id)
                        if len(self.yellow_id) < 3:
                                self.yellow_id[id] = n
                        else:
                            continue
                    idr = self.yellow_id[id]
                    self.state.robots_yellow[idr].pose.x = x
                    self.state.robots_yellow[idr].pose.y = 130 - y
                    self.state.robots_yellow[idr].pose.yaw = -ang
                    #print("myteam:", idr, id, x, y, -ang)

                # print("id:", idr, id, x, y, -ang)
            elif 99 < id < 109:  # Blue Team
                if self.action_manager.is_yellow:
                    if id not in self.yellow_id:
                        n = len(self.yellow_id)
                        if len(self.yellow_id) < 3:
                                self.yellow_id[id] = n
                        else:
                            continue
                    idr = self.yellow_id[id]
                    self.state.robots_yellow[idr].pose.x = x
                    self.state.robots_yellow[idr].pose.y = 130 - y
                    self.state.robots_yellow[idr].pose.yaw = -ang
                    #print("myteam:", idr, id, x, y, -ang)
            else:
                if op_id < 3:
                    self.state.robots_blue[op_id].pose.x = x
                    self.state.robots_blue[op_id].pose.y = 130 - y
                    self.state.robots_blue[op_id].pose.yaw = -ang
                    #print("op:", op_id, id, x, y, -ang)
                    op_id+=1

        self.state.goals_blue = self.goal_left
        self.state.goals_yellow = self.goal_right

        # if debug:
        #     print("robots blue: ")
        #     print(self.state.robots_blue)
        #     print("robots yellow: ")
        #     #print(self.state.robots_yellow)

        # Goal detection
        if not found_ball:  # lost ball (goal?)
            if self.prev_ball is not None and 45 < self.prev_ball.pose.y < 85:
                if self.prev_ball.pose.x > 156:  # yes, left goal
                    self.goal_left += 1
                    self.prev_ball = None
                    print("GOAL LEFT! score: %d x %d" % (self.goal_left, self.goal_right))
                elif self.prev_ball.pose.x < 14:  # yes, right goal
                    self.goal_right += 1
                    self.prev_ball = None
                    print("GOAL RIGHT! score: %d x %d" % (self.goal_left, self.goal_right))

        elif self.prev_ball is not None or 22 < self.state.balls[0].pose.x < 146:
            self.prev_ball = self.state.balls[0]  # only enable a new goal if the ball is seen far from goal again

        #print("state:",self.state.robots_yellow[0].pose.x)
        return self.state

