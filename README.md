# DIGIT_ROS
 ROS Interface for The Digit Sensor
 
 
 It contains 2 nodes for reading data from digit sensors, one for single DIGIT, the other with Two Digits.
 It contains also a service to perform inference.


## Launching a Digit

```roslaunch digit_ros DIGIT.launch ```

Remember to edit te the DIGIT Serial Number found Behind the sensor

## Launching 2 Digits 

```roslaunch digit_ros DIGIT_2_sensors.launch```
    

Remember to edit te the DIGIT Serial Number found Behind the sensor

## Launching the Inference Service

- In one terminal, type: 

```   
source devel/setup.bash
roscore 
```

- In a second termial, type: 

```
source devel/setup.bash
rosrun digit_ros inference_server.py
``` 

- You can test calling the service with:

```
source devel/setup.bash
rosservice call /stability_prediction 1
```
