# -*- coding: utf-8 -*-
"""
@Brief: This is a vision module(single robot) for RoboCup Small Size League
@Version: RoboCup Small Size League 2018
@author: Wang Yunkai
"""

import socket
from time import sleep
import math
from proto.grSim_Packet_pb2 import grSim_Packet
import proto.messages_robocup_ssl_wrapper_pb2 as messages_wrapper
import proto.zss_cmd_pb2 as zss

# MULTI_GROUP = '224.5.23.2'
# VISION_PORT = 10020
# ROBOT_ID = 6

class VisionModule:
    def __init__(self, MULTI_GROUP, VISION_PORT, STATUS_PORT, SENDERIP = '0.0.0.0'):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        self.sock.bind((SENDERIP,VISION_PORT))
        self.sock.setsockopt(socket.IPPROTO_IP,
            socket.IP_ADD_MEMBERSHIP,
            socket.inet_aton(MULTI_GROUP) + socket.inet_aton(SENDERIP))
        
        
        # self.status_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        # self.status_sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        # self.status_sock.bind(('127.0.0.1',STATUS_PORT))
        # self.status_sock.setsockopt(socket.IPPROTO_IP,socket.IP_ADD_MEMBERSHIP,socket.inet_aton(MULTI_GROUP) + socket.inet_aton(SENDERIP))
        # self.status_sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        # self.status_sock.settimeout(0.5)

        self.status_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.status_sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        self.status_sock.bind((SENDERIP,STATUS_PORT))
        self.status_sock.setsockopt(socket.IPPROTO_IP,
            socket.IP_ADD_MEMBERSHIP,
            socket.inet_aton(MULTI_GROUP) + socket.inet_aton(SENDERIP))
        self.status_sock.settimeout(0.2)

        self.robot_info = [0, 0, 0, 0]
        self.ball_info = [0, 0]

    def receive(self):
        print('receive')
        data, addr = self.sock.recvfrom(1024)
        # print(data)
        # sleep(0.001) # wait for reading
        # print('receive2')
        try:
            robot_Data, addr = self.status_sock.recvfrom(1024)
            print('receive2')
        except:         
            robot_Data = None
        return data, robot_Data

    def get_info(self, ROBOT_ID):
        data, robot_Data = self.receive()
        package = messages_wrapper.SSL_WrapperPacket()
        try:
            package.ParseFromString(data)
            
            detection = package.detection
            #print('camera id:', detection.camera_id)
            
            robots = detection.robots_blue # repeat
            robot_max_conf = 0
            for robot in robots:
                if robot.robot_id == ROBOT_ID and robot.confidence > robot_max_conf:
                    self.robot_info[0] = robot.x/1000.0
                    self.robot_info[1] = robot.y/1000.0
                    self.robot_info[2] = robot.orientation
                    #print('Robot', robot.confidence)
            
            balls = detection.balls # repeat
            ball_max_conf = 0
            for ball in balls:
                if ball.confidence >= ball_max_conf:
                    self.ball_info[0] = ball.x/1000.0
                    self.ball_info[1] = ball.y/1000.0

        except:
            print("vision error")
            pass
                #print('Ball', ball.confidence)
        package = zss.Robot_Status
        try:
            package.ParseFromString(robot_Data)
            for robot in package:
                if robot.robot_id == ROBOT_ID:
                    self.robot_info[3] = robot.indrared
        except:
            print("status error")
            pass
  
if __name__ == '__main__':
    MULTI_GROUP = '224.5.23.2'
    VISION_PORT = 10094
    STATUS_PORT = 30011
    ROBOT_ID = 6
    vision = VisionModule(MULTI_GROUP, VISION_PORT, STATUS_PORT)
    while True:
        vision.get_info(ROBOT_ID)
        print(vision.robot_info)
        # print(vision.ball_info)
    print(vision.robot_info)
        