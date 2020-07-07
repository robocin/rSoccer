
class Pose:
    def __init__(self, x=0, y=0, yaw=0):
        self.x = x
        self.y = y
        self.yaw = yaw


class EntityState:
    def __init__(self):
        self.pose = Pose()
        self.v_pose = Pose()


class ProtoState:
    def __init__(self):
        self.balls = []
        self.robots_yellow = []
        self.robots_blue = []
