
from digit_interface import Digit
import cv2
import time
import numpy as np

def compute_baseline(baseline_images):
    num_images = len(baseline_images)
    print(num_images)
    canvas = np.zeros(baseline_images[0].shape)
    for image in baseline_images:
        canvas += image
    canvas /= num_images
    baseline_image = canvas.astype(np.uint8)
    baseline_image = cv2.GaussianBlur(baseline_image,(11,11),0)
    return baseline_image

def recompute_baseline(sensor):
    sensor["baseline"] = []
    # collecting a set of images to denoise and average into a baseline
    start_time = time.time()
    while (time.time() - start_time <  1.5):
        sensor["baseline"].append(sensor["object"].get_frame())
    sensor = compute_baseline(sensor)
    return sensor

def initialize_sensors(sensors,sensor_amesensor_serial_number_list, sensor_name_list, fps = Digit.STREAMS["QVGA"]["fps"]["60fps"], intensity =10):
    for sn, name  in zip(sensor_amesensor_serial_number_list, sensor_name_list):      
        sensor = {}
        sensor["name"] = name
        sensor["serial_number"] = sn
        sensor["object"] = Digit(sn, name)
        # initialize the connection and config
        sensor["object"].connect()
        sensor["object"].set_fps(fps)
        sensor["object"].set_intensity(intensity)
        sensor["baseline"] = []
        sensor["contact_center"] = None
        
        sensors[name] = sensor

    # wait for the light to fully initialize
    start_time = time.time()
    while (time.time() - start_time <  2):
        continue
    # collecting a set of images to denoise and average into a baseline
    start_time = time.time()
    while (time.time() - start_time <  2):
        for _, sensor in sensors.items():
            sensor["baseline"].append(sensor["object"].get_frame())
    
    return sensors