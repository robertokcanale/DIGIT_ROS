
import cv2
import numpy as np
from utils.contact_area_functions import *

def compute_baseline(baseline):
    num_images = len(baseline)
    canvas = np.zeros(baseline[0].shape)
    for image in baseline:
        canvas += image
    canvas /= num_images
    baseline = canvas.astype(np.uint8)
    baseline = cv2.GaussianBlur(baseline,(11,11),0)
    return baseline

def compute_diff(curr, base):
    diff, res = contact_area(target=curr.copy(),base=base)
    if not res is None:
        poly, (major_axis, major_axis_end), (minor_axis, minor_axis_end), center = res
        output = draw_major_minor(cv2.cvtColor(curr.copy(), cv2.COLOR_GRAY2BGR), poly, major_axis, major_axis_end, minor_axis, minor_axis_end)
    else:
        output = curr
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    diff = (diff * 8. * 255)
    diff = np.clip(diff, 0., 255.)
    diff = diff.astype(np.uint8)
                        
    return diff, res, output