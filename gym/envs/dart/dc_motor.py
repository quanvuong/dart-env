import numpy as np

class DCMotor:
    def __init__(self, Kts, Rs, voltage, gear_ratio):
        self.Kts = Kts
        self.Rs = Rs
        self.voltage = voltage
        self.gear_ratio = gear_ratio

    def get_torque(self, command, qdot):
        Vemf = np.array(self.Kts * qdot) * self.gear_ratio
        current = np.array((self.voltage * command - Vemf) / self.Rs) * self.gear_ratio
        return current * self.Kts

