import numpy as np

def compute_rotation_from_normal(normal):
    nx, ny, nz = normal
    pitch = np.arctan2(-nx, nz) / np.pi * 180.
    roll = np.arctan2(ny, nz) / np.pi * 180.
    return pitch, roll
