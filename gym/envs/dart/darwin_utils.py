import numpy as np

def VAL2RADIAN(val):
    return (val - 2048) * 0.088 * np.pi / 180

def RADIAN2VAL(rad):
    return rad * 180 / np.pi / 0.088 + 2048

JOINT_LOW_BOUND = np.array([
- np.pi, - np.pi/2, - np.pi/4, # LEFT HAND
- np.pi/2, - np.pi/8, - np.pi/4, # RIGHT HAND
    -0.01, -0.01, # HEAD
-np.pi / 10, -0.20, -0.25, -0.60, -0.40, -0.40, # LEFT LEG
-np.pi / 10, -0.15, -0.40, -0.40, -0.25, -0.30  # RIGHT LEG
])


JOINT_UP_BOUND = np.array([
np.pi/2, np.pi/8, np.pi/4, # LEFT HAND
np.pi, np.pi/2, np.pi/4, # RIGHT HAND
    0.01, 0.01, # HEAD
np.pi / 10, 0.15, 0.40, 0.40, 0.25, 0.30, # LEFT LEG
np.pi / 10, 0.20, 0.25, 0.60, 0.40, 0.40  # RIGHT LEG
])


