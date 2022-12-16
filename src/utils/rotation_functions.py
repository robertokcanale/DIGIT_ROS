import numpy as np

def contact_center_distance3D(center_1, center_2, gripper_distannce):
    #taking first sensor plane as 0
    #taking second sensor plane as a plane with distance z, and flipping wrt 3D world
    #This is TBD with teerawat
    center_1 = center_1[..., 0]
    center_2[0] = -center_2[0]
    center_2[1] = -center_2[1]
    center_2 = center_1[..., gripper_distannce]
    d = np.linalg.norm(center_2-center_1)   
    return d


def contact_center_distance2D(center_1, center_2):
    d = np.linalg.norm(center_2-center_1)   
    return d

def compute_rotation_from_normal(normal):
    nx, ny, nz = normal
    pitch = np.arctan2(-nx, nz) / np.pi * 180.
    roll = np.arctan2(ny, nz) / np.pi * 180.
    return pitch, roll

def rotation_between2sensors(sensors):
    if (not sensors[0]["contact_center"] is None) and (not sensors[1]["contact_center"] is None):
        pLeft = None
        pRight = None
        for sensor in sensors:
            if "Left" in sensor["name"]:
                pLeft = sensor["contact_center"]
            elif "Right" in sensor["name"]:
                pRight = sensor["contact_center"]
        vi, vj = pRight - pLeft
            # vk = distance from gripper
        vk = 14.0 # dummy value
        normal = np.array((vi, vj, vk))
        normal = normal / (normal**2).sum()**.5
        pitch, roll = compute_rotation_from_normal(normal)

        print("roll:",roll/np.pi*180)
        print("pitch:",pitch/np.pi*180)
            
        tactile_center = np.array(sensors[0]["baseline"].shape[:2]) / 2.
        dx = tactile_center[1] - pLeft[0]
        dy = tactile_center[0] - pLeft[1]
            # dx, dy = tactile_center - pLeft
        print("translation x:",dx)
        print("translation y:",dy)
        print("-----")