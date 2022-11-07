
from digit_interface import Digit
import cv2
import numpy as np
from utils.contact_area_functions import *
from utils.sensor_functions import *

SENSOR_SERIAL_NUMBER_LIST = ["D20492"]
SENSOR_NAME_LIST = ["LeftGripper"]
FPS_CONFIG = Digit.STREAMS["QVGA"]["fps"]["60fps"]
DIGIT_INTENSITY = 10


def main():
    sensors = {}
    sensors = initialize_sensors(sensors, SENSOR_SERIAL_NUMBER_LIST, SENSOR_NAME_LIST)
            
    # compiling set of images into a baseline
    for _, sensor in sensors.items():
        sensor = compute_baseline(sensor)
            

    # digit_pt = pt.PyTouch(pt.sensors.DigitSensor, tasks=[pt.tasks.ContactArea])
    
    while True:
        output_images_list = []
        for _, sensor in sensors.items():
            sensor["current_frame"] = cv2.GaussianBlur(sensor["object"].get_frame(),(11,11),5)
            
            baseline = cv2.cvtColor(sensor["baseline"], cv2.COLOR_BGR2LAB)
            base_l,base_a,base_b = cv2.split(baseline)
            
            current_frame = cv2.cvtColor(sensor["current_frame"], cv2.COLOR_BGR2LAB)
            curr_l,curr_a,curr_b = cv2.split(current_frame)
            
            image_diff_norm = abs(base_b - curr_b)**2.
            image_diff_norm = (image_diff_norm - image_diff_norm.min()) / (image_diff_norm.max() - image_diff_norm.min()) * 255.
            image_diff_norm = image_diff_norm.astype(np.uint8)
            
            raw_stream = contact_area(target=curr_b.copy(),base=base_b)
            diff, res = raw_stream
            if not res is None:
                poly, (major_axis, major_axis_end), (minor_axis, minor_axis_end), center = res
                print("Contact Center", center)

                output = draw_major_minor(cv2.cvtColor(curr_b.copy(), cv2.COLOR_GRAY2BGR), poly, major_axis, major_axis_end, minor_axis, minor_axis_end)
            else:
                output = curr_b
                
                output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
           
            diff = (diff * 8. * 255)
            diff = np.clip(diff, 0., 255.)
            diff = diff.astype(np.uint8)
            
            final_visualized_image = np.hstack((base_b, curr_b, diff))
            final_visualized_image = cv2.cvtColor(final_visualized_image, cv2.COLOR_GRAY2BGR)
            final_visualized_image = np.hstack((final_visualized_image, output))
                       
        # final_visualized_image = np.vstack(output_images_list)
        cv2.imshow("Output", final_visualized_image)
        if cv2.waitKey(1) == 27:
            for sensor in sensors:
                sensor.disconnect()
            break
        
if __name__ == '__main__':
    main()