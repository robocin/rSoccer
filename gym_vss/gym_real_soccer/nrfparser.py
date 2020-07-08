import os
import sys
import time

path = '/'.join(os.path.abspath(__file__).split('/')[:-1])
sys.path.append(path)

import _nrf_Communication as nrf


class NRFParser:
    def __init__(self):
        self.comm = nrf.new_nrf_Communication()
        nrf.nrf_Communication_setup(self.comm)
    
    def send_speeds(self, left_speed, right_speed, idx):
        nrf.nrf_Communication_sendSpeed(self.comm, idx, int(left_speed), int(right_speed), int(0))
        time.sleep(0.0015)
        # time.sleep(0.0006)
