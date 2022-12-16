import robotiq_2f_gripper_control.baseRobotiq2FGripper
import robotiq_modbus_rtu.comModbusRtu
from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_output  as outputMsg

def connect_robotiq_gripper(device):
    gripper = robotiq_2f_gripper_control.baseRobotiq2FGripper.robotiqbaseRobotiq2FGripper()
    gripper.client = robotiq_modbus_rtu.comModbusRtu.communication()
    gripper.client.connectToDevice(device)
    command = outputMsg()
    #Reset
    command.rACT = 0
    gripper.refreshCommand(command)
    gripper.sendCommand()
    #Activate
    command.rACT = 1
    command.rGTO = 1
    command.rSP  = 255
    command.rFR  = 150
    gripper.refreshCommand(command)
    gripper.sendCommand()
    return gripper, command