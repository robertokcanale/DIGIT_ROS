
from digit_interface import Digit
import cv2
import time
import numpy as np

# def compute_baseline(sensor):
#     num_images = len(sensor["baseline"])

#     canvas = np.zeros(sensor["baseline"][0].shape)
#     for image in sensor["baseline"]:
#         canvas += image
#     canvas /= num_images
#     sensor["baseline"] = canvas.astype(np.uint8)
#     sensor["baseline"] = cv2.GaussianBlur(sensor["baseline"],(11,11),0)
#     return sensor

def compute_baseline(baseline):
    num_images = len(baseline)
    canvas = np.zeros(baseline[0].shape)
    for image in baseline:
        canvas += image
    canvas /= num_images
    baseline = canvas.astype(np.uint8)
    baseline = cv2.GaussianBlur(baseline,(11,11),0)
    return baseline

def initialize_sensor(sensor_serial_number, sensor_name, fps = Digit.STREAMS["QVGA"]["fps"]["60fps"], intensity = 10):
    sensor = {}
    sensor["name"] = sensor_name
    sensor["serial_number"] = sensor_serial_number
    sensor["object"] = Digit(sensor_serial_number, sensor_name)
    # initialize the connection and config
    sensor["object"].connect()
    sensor["object"].set_fps(fps)
    sensor["object"].set_intensity(intensity)
    sensor["baseline"] = []
    sensor["contact_center"] = None

    # wait for the light to fully initialize
    start_time = time.time()
    while (time.time() - start_time <  2):
        continue
    # collecting a set of images to denoise and average into a baseline
    start_time = time.time()
    while (time.time() - start_time <  2):
        sensor["baseline"].append(sensor["object"].get_frame())
    sensor = compute_baseline(sensor)
    return sensor

def initialize_sensors(sensors,sensor_serial_number_list, sensor_name_list, fps = Digit.STREAMS["QVGA"]["fps"]["60fps"], intensity = 10):
    for sn, name  in zip(sensor_serial_number_list, sensor_name_list):      
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