

class ActionManager(object):

    _instance = None

    def __init__(self):
        self.has_event = False
        self.event_type = None

        self.is_paused = True
        self.is_yellow = True

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ActionManager, cls).__new__(cls, *args, **kwargs)

        return cls._instance

    def pause(self):
        self.is_paused = True
        self.has_event = True
        self.event_type = 'p'

    def resume(self):
        self.is_paused = False
        self.has_event = True
        self.event_type = 'p'

    def setTeamYellow(self):
        self.is_yellow = True
        self.has_event = True
        self.event_type = 'c'

    def setTeamBlue(self):
        self.is_yellow = False
        self.has_event = True
        self.event_type = 'c'

    def stopRobots(self):
        self.has_event = True
        self.event_type = 's'

    def clearEvent(self):
        self.has_event = False
        self.event_type = None





