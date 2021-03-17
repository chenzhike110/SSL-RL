# -*- coding: utf-8 -*-
"""
@Brief: This is an action module to control grSim's robots
"""

# from os import replace
from math import sin
import socket
# import struct
from time import sleep
import math
from random import random

from numpy.matrixlib.defmatrix import mat
import proto.grSim_Packet_pb2 as sim_pkg

class ActionModule:
    def __init__(self, ACTION_IP, ACTION_PORT):
        self.address = (ACTION_IP, ACTION_PORT)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    def send_start_package(self):
        self.socket.sendto(self.start_package, self.address)
    
    def send_action(self, robot_num=0, vx=0, vy=0, w=0, kp=0):
        package = sim_pkg.grSim_Packet()
        commands = package.commands
        
        commands.timestamp = 0
        commands.isteamyellow = False
        command = commands.robot_commands.add()
        command.id = robot_num
        command.kickspeedx = kp
        command.kickspeedz = 0
        command.veltangent = vx
        command.velnormal = vy
        command.velangular = w
        command.spinner = 1
        command.wheelsspeed = False
        
        self.socket.sendto(package.SerializeToString(), self.address)
    
    def send_reset(self, robot_num=0, ifrandom=0):
        package = sim_pkg.grSim_Packet()
        replacement = package.replacement
        if ifrandom:
            x = 2*(random()-0.5)*4.0
            y = 2*(random()-0.5)*2.5
            theta = 4*(random()-0.5)*math.pi
            ball = replacement.ball
            ball.x = x+0.105
            ball.y = y
            ball.vx = 0.0
            ball.vy = 0.0
        for i in range(16):
            robot = replacement.robots.add()
            if i != robot_num:
                robot.x = i*0.3
                robot.y = -5.0
                robot.id = i
                robot.dir = 0
                robot.yellowteam = False
                robot.turnon = True 
            else:
                if ifrandom == 0:
                    robot.x = i*0.3
                    robot.y = 0.0
                    robot.id = i
                    robot.dir = 0.0
                    robot.yellowteam = False
                    robot.turnon = True
                else:
                    robot.x = x
                    robot.y = y
                    robot.id = i
                    robot.dir = theta
                    robot.yellowteam = False
                    robot.turnon = True
            robot = replacement.robots.add()
            robot.x = -i*0.3
            robot.y = -5.5
            robot.id = i
            robot.dir = 0
            robot.yellowteam = True
            robot.turnon = True
        
        self.socket.sendto(package.SerializeToString(), self.address)
        
    def reset(self, robot_num):
        package = sim_pkg.grSim_Packet()
        replacement = package.replacement
        #ball_rep = replacement.ball
        bot_rep = replacement.robots.add()
        
        #ball_rep.x = 100.0
        #ball_rep.y = 0.0
        #ball_rep.vx = 0.0
        #ball_rep.vy = 0.0
        
        bot_rep.x = 0.0
        bot_rep.y = 0.0
        bot_rep.dir = 0.0
        bot_rep.id = robot_num
        bot_rep.yellowteam = False
        
        self.socket.sendto(package.SerializeToString(), self.address)
        
if __name__ == "__main__":
    ACTION_IP = '127.0.0.1' # local host grSim
    ACTION_PORT = 20011 # grSim command listen port
    action = ActionModule(ACTION_IP, ACTION_PORT)
    # action.send_reset(robot_num=6, ifrandom=1)
    while(True):
        action.send_action(robot_num=6, vx=0, vy=0, w=0.1, kp=1)
        sleep(0.015)