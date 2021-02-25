import copy
from mmpose.apis import (inference_top_down_pose_model,
                         init_pose_model, vis_pose_result)
from mmdet.apis import inference_detector, init_detector
import glob
import numpy as np
import datetime
import cv2
import mmcv
from numpy import linalg as LA
import math
import collections
from argparse import ArgumentParser
import json
import time
# from model import manual
from util.socketUtils import socket
import os
import redis

queue_name = "img_live_demo"

img_redis = redis.Redis(host='localhost', port=6379, db=0)
img_sub = img_redis.pubsub()
img_sub.subscribe(queue_name)


# def analyzeAngle(room):
#     print('run analyzeAngle >>>> for >', room)
#     print("analyzeAngleanalyzeAngleanalyzeAngleanalyzeAngle")
#     for msg in img_sub.listen():
#         mess = msg['data']
#         print("mess: ", mess)
#     i = 0
#     result = [
#         [
#             {
#                 "name": "0.png",
#                 "pose": [
#                     {"name": "RKnee", "angle": "0.41"},
#                     {"name": "RHip", "angle": "3.17"},
#                     {"name": "RUperHip", "angle": "2.62"},
#                     {"name": "RShoulder", "angle": "6.51"},
#                     {"name": "RShoulderElbow", "angle": "-2.17"},
#                     {"name": "RElbow", "angle": "9.83"},
#                     {"name": "LKnee", "angle": "-0.66"},
#                     {"name": "LHip", "angle": "5.43"},
#                     {"name": "LUperHip", "angle": "-3.99"},
#                     {"name": "LShoulder", "angle": "-5.14"},
#                     {"name": "LShoulderElbow", "angle": "-10.07"},
#                     {"name": "LElbow", "angle": "10.67"}
#                 ],
#                 "path": "vis_results2021.02.22-09.01.52/1613959339.265736/vis_1613959319.6404662_1.jpg.png"
#             },
#             {
#                 "name": "1.png",
#                 "pose": [
#                     {"name": "RKnee", "angle": "1.93"},
#                     {"name": "RHip", "angle": "-3.07"},
#                     {"name": "RUperHip", "angle": "8.66"},
#                     {"name": "RShoulder", "angle": "-8.13"},
#                     {"name": "RShoulderElbow", "angle": "-7.42"},
#                     {"name": "RElbow", "angle": "-3.07"},
#                     {"name": "LKnee", "angle": "5.36"},
#                     {"name": "LHip", "angle": "5.62"},
#                     {"name": "LUperHip", "angle": "-9.72"},
#                     {"name": "LShoulder", "angle": "9.19"},
#                     {"name": "LShoulderElbow", "angle": "-13.84"},
#                     {"name": "LElbow", "angle": "34.42"}
#                 ],
#                 "path": "vis_results2021.02.22-09.01.52/1613959339.265736/vis_1613959319.6404662_2.jpg.png"
#             },
#             {
#                 "name": "2.png",
#                 "pose": [
#                     {"name": "RKnee", "angle": "-2.66"},
#                     {"name": "RHip", "angle": "4.09"},
#                     {"name": "RUperHip", "angle": "1.98"},
#                     {"name": "RShoulder", "angle": "-9.99"},
#                     {"name": "RShoulderElbow", "angle": "-5.08"},
#                     {"name": "RElbow", "angle": "-2.41"},
#                     {"name": "LKnee", "angle": "11.68"},
#                     {"name": "LHip", "angle": "-1.23"},
#                     {"name": "LUperHip", "angle": "-5.4"},
#                     {"name": "LShoulder", "angle": "13.41"},
#                     {"name": "LShoulderElbow", "angle": "0.75"},
#                     {"name": "LElbow", "angle": "16.43"}
#                 ],
#                 "path": "vis_results2021.02.22-09.01.52/1613959339.265736/vis_1613959319.6404662_3.jpg.png"
#             },
#             {
#                 "name": "3.png",
#                 "pose": [
#                     {"name": "RKnee", "angle": "-6.53"},
#                     {"name": "RHip", "angle": "4.21"},
#                     {"name": "RUperHip", "angle": "6.31"},
#                     {"name": "RShoulder", "angle": "-16.44"},
#                     {"name": "RShoulderElbow", "angle": "47.19"},
#                     {"name": "RElbow", "angle": "-78.17"},
#                     {"name": "LKnee", "angle": "12.33"},
#                     {"name": "LHip", "angle": "-4.68"},
#                     {"name": "LUperHip", "angle": "-8.3"},
#                     {"name": "LShoulder", "angle": "18.43"},
#                     {"name": "LShoulderElbow", "angle": "-11.05"},
#                     {"name": "LElbow", "angle": "54.01"}
#                 ],
#                 "path": "vis_results2021.02.22-09.01.52/1613959339.265736/vis_1613959319.6404662_4.jpg.png"
#             },
#             {
#                 "name": "4.png",
#                 "pose": [
#                     {"name": "RKnee", "angle": "9.14"},
#                     {"name": "RHip", "angle": "-4.54"},
#                     {"name": "RUperHip", "angle": "18.13"},
#                     {"name": "RShoulder", "angle": "-6.97"},
#                     {"name": "RShoulderElbow", "angle": "-27.2"},
#                     {"name": "RElbow", "angle": "30.19"},
#                     {"name": "LKnee", "angle": "-1.53"},
#                     {"name": "LHip", "angle": "6.28"},
#                     {"name": "LUperHip", "angle": "-16.78"},
#                     {"name": "LShoulder", "angle": "5.61"},
#                     {"name": "LShoulderElbow", "angle": "-6.31"},
#                     {"name": "LElbow", "angle": "15.12"}
#                 ],
#                 "path": "vis_results2021.02.22-09.01.52/1613959339.265736/vis_1613959319.6404662_5.jpg.png"
#             },
#             {
#                 "name": "5.png",
#                 "pose": [
#                     {"name": "RKnee", "angle": "15.97"},
#                     {"name": "RHip", "angle": "3.01"},
#                     {"name": "RUperHip", "angle": "33.09"},
#                     {"name": "RShoulder", "angle": "-16.92"},
#                     {"name": "RShoulderElbow", "angle": "14.36"},
#                     {"name": "RElbow", "angle": "-21.63"},
#                     {"name": "LKnee", "angle": "10.31"},
#                     {"name": "LHip", "angle": "-13.91"},
#                     {"name": "LUperHip", "angle": "-30.78"},
#                     {"name": "LShoulder", "angle": "14.61"},
#                     {"name": "LShoulderElbow", "angle": "-6.41"},
#                     {"name": "LElbow", "angle": "-14.02"}
#                 ],
#                 "path": "vis_results2021.02.22-09.01.52/1613959339.265736/vis_1613959319.6404662_6.jpg.png"
#             },
#             {
#                 "name": "6.png",
#                 "pose": [
#                     {"name": "RKnee", "angle": "19.61"},
#                     {"name": "RHip", "angle": "13.15"},
#                     {"name": "RUperHip", "angle": "19.76"},
#                     {"name": "RShoulder", "angle": "-18.62"},
#                     {"name": "RShoulderElbow", "angle": "3.14"},
#                     {"name": "RElbow", "angle": "7.62"},
#                     {"name": "LKnee", "angle": "2.63"},
#                     {"name": "LHip", "angle": "-24.95"},
#                     {"name": "LUperHip", "angle": "-15.22"},
#                     {"name": "LShoulder", "angle": "14.08"},
#                     {"name": "LShoulderElbow", "angle": "4.47"},
#                     {"name": "LElbow", "angle": "-30.49"}
#                 ],
#                 "path": "vis_results2021.02.22-09.01.52/1613959339.265736/vis_1613959319.6404662_7.jpg.png"
#             },
#             {
#                 "name": "7.png",
#                 "pose": [
#                     {"name": "RKnee", "angle": "6.44"},
#                     {"name": "RHip", "angle": "32.44"},
#                     {"name": "RUperHip", "angle": "-33.37"},
#                     {"name": "RShoulder", "angle": "-27.04"},
#                     {"name": "RShoulderElbow", "angle": "-39.08"},
#                     {"name": "RElbow", "angle": "88.73"},
#                     {"name": "LKnee", "angle": "7.7"},
#                     {"name": "LHip", "angle": "-52.91"},
#                     {"name": "LUperHip", "angle": "33.93"},
#                     {"name": "LShoulder", "angle": "26.48"},
#                     {"name": "LShoulderElbow", "angle": "15.09"},
#                     {"name": "LElbow", "angle": "-43.3"}
#                 ],
#                 "path": "vis_results2021.02.22-09.01.52/1613959339.265736/vis_1613959319.6404662_8.jpg.png"
#             }
#         ],
#         [
#             {
#                 "name": "0.png",
#                 "pose": [
#                     {"name": "RKnee", "angle": "4.74"},
#                     {"name": "RHip", "angle": "-3.71"},
#                     {"name": "RUperHip", "angle": "5.3"},
#                     {"name": "RShoulder", "angle": "6.13"},
#                     {"name": "RShoulderElbow", "angle": "-1.15"},
#                     {"name": "RElbow", "angle": "-2.05"},
#                     {"name": "LKnee", "angle": "-3.22"},
#                     {"name": "LHip", "angle": "10.9"},
#                     {"name": "LUperHip", "angle": "-11.63"},
#                     {"name": "LShoulder", "angle": "0.2"},
#                     {"name": "LShoulderElbow", "angle": "-2.72"},
#                     {"name": "LElbow", "angle": "3.53"}
#                 ],
#                 "path": "vis_results2021.02.22-09.01.52/1613959351.1049576/vis_1613959319.6404662_1.jpg.png"
#             },
#             {
#                 "name": "1.png",
#                 "pose": [
#                     {"name": "RKnee", "angle": "-0.94"},
#                     {"name": "RHip", "angle": "0.35"},
#                     {"name": "RUperHip", "angle": "4.06"},
#                     {"name": "RShoulder", "angle": "-5.24"},
#                     {"name": "RShoulderElbow", "angle": "3.67"},
#                     {"name": "RElbow", "angle": "-8.4"},
#                     {"name": "LKnee", "angle": "6.48"},
#                     {"name": "LHip", "angle": "4.72"},
#                     {"name": "LUperHip", "angle": "-6.77"},
#                     {"name": "LShoulder", "angle": "7.95"},
#                     {"name": "LShoulderElbow", "angle": "-20.68"},
#                     {"name": "LElbow", "angle": "42.33"}
#                 ],
#                 "path": "vis_results2021.02.22-09.01.52/1613959351.1049576/vis_1613959319.6404662_2.jpg.png"
#             },
#             {
#                 "name": "2.png",
#                 "pose": [
#                     {"name": "RKnee", "angle": "0.08"},
#                     {"name": "RHip", "angle": "-0.56"},
#                     {"name": "RUperHip", "angle": "4.46"},
#                     {"name": "RShoulder", "angle": "-5.65"},
#                     {"name": "RShoulderElbow", "angle": "0.12"},
#                     {"name": "RElbow", "angle": "6.42"},
#                     {"name": "LKnee", "angle": "8.5"},
#                     {"name": "LHip", "angle": "3.78"},
#                     {"name": "LUperHip", "angle": "-5.66"},
#                     {"name": "LShoulder", "angle": "6.85"},
#                     {"name": "LShoulderElbow", "angle": "-3.83"},
#                     {"name": "LElbow", "angle": "24.1"}
#                 ],
#                 "path": "vis_results2021.02.22-09.01.52/1613959351.1049576/vis_1613959319.6404662_3.jpg.png"
#             },
#             {
#                 "name": "3.png",
#                 "pose": [
#                     {"name": "RKnee", "angle": "-0.49"},
#                     {"name": "RHip", "angle": "11.21"},
#                     {"name": "RUperHip", "angle": "-3.27"},
#                     {"name": "RShoulder", "angle": "-17.82"},
#                     {"name": "RShoulderElbow", "angle": "-2.54"},
#                     {"name": "RElbow", "angle": "-34.05"},
#                     {"name": "LKnee", "angle": "17.04"},
#                     {"name": "LHip", "angle": "-13.61"},
#                     {"name": "LUperHip", "angle": "0.64"},
#                     {"name": "LShoulder", "angle": "20.45"},
#                     {"name": "LShoulderElbow", "angle": "-8.42"},
#                     {"name": "LElbow", "angle": "60.86"}
#                 ],
#                 "path": "vis_results2021.02.22-09.01.52/1613959351.1049576/vis_1613959319.6404662_4.jpg.png"
#             },
#             {
#                 "name": "4.png",
#                 "pose": [
#                     {"name": "RKnee", "angle": "5.84"},
#                     {"name": "RHip", "angle": "-1.31"},
#                     {"name": "RUperHip", "angle": "13.82"},
#                     {"name": "RShoulder", "angle": "-3.58"},
#                     {"name": "RShoulderElbow", "angle": "-14.09"},
#                     {"name": "RElbow", "angle": "24.12"},
#                     {"name": "LKnee", "angle": "-0.12"},
#                     {"name": "LHip", "angle": "6.03"},
#                     {"name": "LUperHip", "angle": "-13.99"},
#                     {"name": "LShoulder", "angle": "3.74"},
#                     {"name": "LShoulderElbow", "angle": "-14.65"},
#                     {"name": "LElbow", "angle": "23.24"}
#                 ],
#                 "path": "vis_results2021.02.22-09.01.52/1613959351.1049576/vis_1613959319.6404662_5.jpg.png"
#             },
#             {
#                 "name": "5.png",
#                 "pose": [
#                     {"name": "RKnee", "angle": "17.58"},
#                     {"name": "RHip", "angle": "6.13"},
#                     {"name": "RUperHip", "angle": "29.45"},
#                     {"name": "RShoulder", "angle": "-24.29"},
#                     {"name": "RShoulderElbow", "angle": "19.82"},
#                     {"name": "RElbow", "angle": "3.24"},
#                     {"name": "LKnee", "angle": "2.49"},
#                     {"name": "LHip", "angle": "-18.37"},
#                     {"name": "LUperHip", "angle": "-16.08"},
#                     {"name": "LShoulder", "angle": "10.93"},
#                     {"name": "LShoulderElbow", "angle": "-14.88"},
#                     {"name": "LElbow", "angle": "-36.07"}
#                 ],
#                 "path": "vis_results2021.02.22-09.01.52/1613959351.1049576/vis_1613959319.6404662_6.jpg.png"
#             },
#             {
#                 "name": "6.png",
#                 "pose": [
#                     {"name": "RKnee", "angle": "17.89"},
#                     {"name": "RHip", "angle": "13.09"},
#                     {"name": "RUperHip", "angle": "20.54"},
#                     {"name": "RShoulder", "angle": "-17.43"},
#                     {"name": "RShoulderElbow", "angle": "7.49"},
#                     {"name": "RElbow", "angle": "5.09"},
#                     {"name": "LKnee", "angle": "-0.94"},
#                     {"name": "LHip", "angle": "-21.98"},
#                     {"name": "LUperHip", "angle": "-13.51"},
#                     {"name": "LShoulder", "angle": "10.4"},
#                     {"name": "LShoulderElbow", "angle": "-1.67"},
#                     {"name": "LElbow", "angle": "-31.29"}
#                 ],
#                 "path": "vis_results2021.02.22-09.01.52/1613959351.1049576/vis_1613959319.6404662_7.jpg.png"
#             },
#             {
#                 "name": "7.png",
#                 "pose": [
#                     {"name": "RKnee", "angle": "18.67"},
#                     {"name": "RHip", "angle": "12.96"},
#                     {"name": "RUperHip", "angle": "-13.57"},
#                     {"name": "RShoulder", "angle": "3.94"},
#                     {"name": "RShoulderElbow", "angle": "-20.43"},
#                     {"name": "RElbow", "angle": "81.21"},
#                     {"name": "LKnee", "angle": "8.03"},
#                     {"name": "LHip", "angle": "-37.29"},
#                     {"name": "LUperHip", "angle": "5.95"},
#                     {"name": "LShoulder", "angle": "3.68"},
#                     {"name": "LShoulderElbow", "angle": "17.32"},
#                     {"name": "LElbow", "angle": "-78.49"}
#                 ],
#                 "path": "vis_results2021.02.22-09.01.52/1613959351.1049576/vis_1613959319.6404662_8.jpg.png"
#             }
#         ],
#         [
#             {
#                 "name": "0.png",
#                 "pose": [
#                     {"name": "RKnee", "angle": "5.66"},
#                     {"name": "RHip", "angle": "-1.0"},
#                     {"name": "RUperHip", "angle": "8.4"},
#                     {"name": "RShoulder", "angle": "5.57"},
#                     {"name": "RShoulderElbow", "angle": "-2.22"},
#                     {"name": "RElbow", "angle": "-1.78"},
#                     {"name": "LKnee", "angle": "3.04"},
#                     {"name": "LHip", "angle": "7.32"},
#                     {"name": "LUperHip", "angle": "-12.82"},
#                     {"name": "LShoulder", "angle": "-1.15"},
#                     {"name": "LShoulderElbow", "angle": "-1.91"},
#                     {"name": "LElbow", "angle": "3.48"}
#                 ],
#                 "path": "vis_results2021.02.22-09.01.52/1613959365.75451/vis_1613959319.6404662_1.jpg.png"
#             },
#             {
#                 "name": "1.png",
#                 "pose": [
#                     {"name": "RKnee", "angle": "-1.06"},
#                     {"name": "RHip", "angle": "-0.77"},
#                     {"name": "RUperHip", "angle": "7.55"},
#                     {"name": "RShoulder", "angle": "-8.24"},
#                     {"name": "RShoulderElbow", "angle": "-5.72"},
#                     {"name": "RElbow", "angle": "-2.4"},
#                     {"name": "LKnee", "angle": "5.55"},
#                     {"name": "LHip", "angle": "6.37"},
#                     {"name": "LUperHip", "angle": "-10.27"},
#                     {"name": "LShoulder", "angle": "10.97"},
#                     {"name": "LShoulderElbow", "angle": "-18.83"},
#                     {"name": "LElbow", "angle": "39.99"}
#                 ],
#                 "path": "vis_results2021.02.22-09.01.52/1613959365.75451/vis_1613959319.6404662_2.jpg.png"
#             },
#             {
#                 "name": "2.png",
#                 "pose": [
#                     {"name": "RKnee", "angle": "-5.12"},
#                     {"name": "RHip", "angle": "-2.35"},
#                     {"name": "RUperHip", "angle": "9.75"},
#                     {"name": "RShoulder", "angle": "-14.51"},
#                     {"name": "RShoulderElbow", "angle": "-12.81"},
#                     {"name": "RElbow", "angle": "-2.23"},
#                     {"name": "LKnee", "angle": "8.63"},
#                     {"name": "LHip", "angle": "6.98"},
#                     {"name": "LUperHip", "angle": "-13.64"},
#                     {"name": "LShoulder", "angle": "18.4"},
#                     {"name": "LShoulderElbow", "angle": "4.29"},
#                     {"name": "LElbow", "angle": "26.63"}
#                 ],
#                 "path": "vis_results2021.02.22-09.01.52/1613959365.75451/vis_1613959319.6404662_3.jpg.png"
#             },
#             {
#                 "name": "3.png",
#                 "pose": [
#                     {"name": "RKnee", "angle": "-0.16"},
#                     {"name": "RHip", "angle": "4.36"},
#                     {"name": "RUperHip", "angle": "8.19"},
#                     {"name": "RShoulder", "angle": "29.39"},
#                     {"name": "RShoulderElbow", "angle": "65.44"},
#                     {"name": "RElbow", "angle": "-37.67"},
#                     {"name": "LKnee", "angle": "21.81"},
#                     {"name": "LHip", "angle": "-11.7"},
#                     {"name": "LUperHip", "angle": "-0.79"},
#                     {"name": "LShoulder", "angle": "-36.78"},
#                     {"name": "LShoulderElbow", "angle": "-58.99"},
#                     {"name": "LElbow", "angle": "39.67"}
#                 ],
#                 "path": "vis_results2021.02.22-09.01.52/1613959365.75451/vis_1613959319.6404662_4.jpg.png"
#             },
#             {
#                 "name": "4.png",
#                 "pose": [
#                     {"name": "RKnee", "angle": "5.96"},
#                     {"name": "RHip", "angle": "-2.09"},
#                     {"name": "RUperHip", "angle": "16.94"},
#                     {"name": "RShoulder", "angle": "-6.69"},
#                     {"name": "RShoulderElbow", "angle": "-24.41"},
#                     {"name": "RElbow", "angle": "30.53"},
#                     {"name": "LKnee", "angle": "-1.17"},
#                     {"name": "LHip", "angle": "7.17"},
#                     {"name": "LUperHip", "angle": "-17.19"},
#                     {"name": "LShoulder", "angle": "6.95"},
#                     {"name": "LShoulderElbow", "angle": "-11.87"},
#                     {"name": "LElbow", "angle": "20.59"}
#                 ],
#                 "path": "vis_results2021.02.22-09.01.52/1613959365.75451/vis_1613959319.6404662_5.jpg.png"
#             },
#             {
#                 "name": "5.png",
#                 "pose": [
#                     {"name": "RKnee", "angle": "18.62"},
#                     {"name": "RHip", "angle": "2.92"},
#                     {"name": "RUperHip", "angle": "26.74"},
#                     {"name": "RShoulder", "angle": "-16.12"},
#                     {"name": "RShoulderElbow", "angle": "1.49"},
#                     {"name": "RElbow", "angle": "4.01"},
#                     {"name": "LKnee", "angle": "4.68"},
#                     {"name": "LHip", "angle": "-8.28"},
#                     {"name": "LUperHip", "angle": "-21.37"},
#                     {"name": "LShoulder", "angle": "10.75"},
#                     {"name": "LShoulderElbow", "angle": "-13.19"},
#                     {"name": "LElbow", "angle": "-5.91"}
#                 ],
#                 "path": "vis_results2021.02.22-09.01.52/1613959365.75451/vis_1613959319.6404662_6.jpg.png"
#             },
#             {
#                 "name": "6.png",
#                 "pose": [
#                     {"name": "RKnee", "angle": "17.76"},
#                     {"name": "RHip", "angle": "7.91"},
#                     {"name": "RUperHip", "angle": "16.5"},
#                     {"name": "RShoulder", "angle": "-5.89"},
#                     {"name": "RShoulderElbow", "angle": "-4.52"},
#                     {"name": "RElbow", "angle": "6.79"},
#                     {"name": "LKnee", "angle": "2.78"},
#                     {"name": "LHip", "angle": "-11.78"},
#                     {"name": "LUperHip", "angle": "-18.75"},
#                     {"name": "LShoulder", "angle": "8.13"},
#                     {"name": "LShoulderElbow", "angle": "-4.76"},
#                     {"name": "LElbow", "angle": "1.39"}
#                 ],
#                 "path": "vis_results2021.02.22-09.01.52/1613959365.75451/vis_1613959319.6404662_7.jpg.png"
#             },
#             {
#                 "name": "7.png",
#                 "pose": [
#                     {"name": "RKnee", "angle": "19.81"},
#                     {"name": "RHip", "angle": "7.64"},
#                     {"name": "RUperHip", "angle": "-9.66"},
#                     {"name": "RShoulder", "angle": "91.62"},
#                     {"name": "RShoulderElbow", "angle": "52.43"},
#                     {"name": "RElbow", "angle": "76.01"},
#                     {"name": "LKnee", "angle": "6.98"},
#                     {"name": "LHip", "angle": "-26.24"},
#                     {"name": "LUperHip", "angle": "-8.48"},
#                     {"name": "LShoulder", "angle": "286.51"},
#                     {"name": "LShoulderElbow", "angle": "-88.25"},
#                     {"name": "LElbow", "angle": "-12.1"}
#                 ],
#                 "path": "vis_results2021.02.22-09.01.52/1613959365.75451/vis_1613959319.6404662_8.jpg.png"
#             }
#         ]
#     ]

#     while True:
#         print('run analyzeAngle >>>>>')
#         i += 1
#         time.sleep(3)
#         if (i > 3):
#             for item in result:
#                 socket.emit('analyzeAngle', item, room=room)
#                 time.sleep(3)
#             break

dict_angles = [
    ['right_hip', 'right_knee', 'right_ankle'], 
    ['left_hip', 'right_hip', 'right_knee'],
    ['right_shoulder', 'right_hip', 'left_hip'],
    ['left_shoulder', 'right_shoulder', 'right_hip'],
    ['left_shoulder', 'right_shoulder', 'right_elbow'],
    ['right_shoulder', 'right_elbow', 'right_wrist'],
    ['left_ankle', 'left_knee', 'left_hip'],
    ['left_knee', 'left_hip', 'right_hip'],
    ['right_hip', 'left_hip', 'left_shoulder'],
    ['left_hip', 'left_shoulder', 'right_shoulder'],
    ['left_elbow', 'left_shoulder', 'right_shoulder'],
    ['left_wrist', 'left_elbow', 'left_shoulder'],
]

dict_name_angles = [
    "RKnee", 
    "RHip",
    "RUperHip",
    "RShoulder",
    "RShoulderElbow",
    "RElbow",
    "LKnee",
    "LHip",
    "LUperHip",
    "LShoulder",
    "LShoulderElbow",
    "LElbow",
]

classnames = dict({0: 'nose',
        1: 'left_eye',
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle'})

id_classnames = {v: k for k, v in classnames.items()}

palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])

original_skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], 
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11]]
pose_limb_color = palette[[18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18]]
pose_limb_color_correct = palette[[16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]]
pose_kpt_color = palette[[18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18]]
pose_kpt_color_correct = palette[[16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]]

FREQ_START_SPLIT = 20
FREQ_END_SPLIT = 3

threshold = 10
w, h = int(477), int(900)
size = (w*5, h)
not_include = [1, 2, 3, 4, 5]
# frontal view
current_timestamp = str(time.time())
folder_mmpose_path = "/home/ddes-0351/Documents/source_code/mmpose"
folder_path = "/home/ddes-0351/Documents/source_code/mmpose/golf_pose"
folder_pos = os.path.join(folder_path, "positive", "3_3")
folder_neg = os.path.join(folder_path, "temp", current_timestamp) #women, kakao_905
folder_8poses = os.path.join(folder_path, "8poses", current_timestamp) #women, kakao_905

def process_mmdet_results(mmdet_results, cat_id=0):
    """Process mmdet results, and return a list of bboxes.

    :param mmdet_results:
    :param cat_id: category id (default: 0 for human)
    :return: a list of detected bounding boxes
    """
    if isinstance(mmdet_results, tuple):
        det_results = mmdet_results[0]
    else:
        det_results = mmdet_results
    return det_results[cat_id]

def get_angle(a, b, c):
    ang = math.degrees(math.atan2(
        c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

def get_angle_draw(a, b, c):
    ang1 = math.degrees(math.atan2(
        c[1] - b[1], c[0] - b[0]))
    ang1 = ang1 + 360 if ang1 < 0 else ang1
    ang2 = math.degrees(math.atan2(a[1] - b[1], a[0] - b[0]))
    ang2 = ang2 + 360 if ang2 < 0 else ang2
    return ang1, ang2

def pairwise_angles(v1, v2):
    acos_invalue = np.sum(v1*v2, axis=-1) / (LA.norm(v1, axis=-1) * LA.norm(v2, axis=-1))
    acos_value = np.arccos(np.round(acos_invalue, 2))
    degrees_value = np.degrees(acos_value)
    where_are_nans = np.isnan(degrees_value)
    degrees_value[where_are_nans] = 20000
    return degrees_value

def pair_angles(kp, dict_angles):
    kp_comp = np.array([np.array([x, y]) for (x, y, c) in kp['annotations'][0]['keypoints']])
    angles_ = []
    angles_draw_ = []
    angles_coor_ = []
    angles_name_ = []
    kp_angles = [(get_angle(kp_comp[id_classnames[angle_a]], 
                    kp_comp[id_classnames[angle_b]], 
                    kp_comp[id_classnames[angle_c]]),
                get_angle_draw(kp_comp[id_classnames[angle_a]], 
                    kp_comp[id_classnames[angle_b]], 
                    kp_comp[id_classnames[angle_c]]),
                (kp_comp[id_classnames[angle_a]], 
                kp_comp[id_classnames[angle_b]], 
                kp_comp[id_classnames[angle_c]]),
                (angle_a, angle_b, angle_c))
                for angle_a, angle_b, angle_c in dict_angles]
    for angle_a, angle_b, angle_c in dict_angles:
        angles_.append((get_angle(kp_comp[id_classnames[angle_a]], 
                    kp_comp[id_classnames[angle_b]], 
                    kp_comp[id_classnames[angle_c]])))
        angles_draw_.append(
            get_angle_draw(kp_comp[id_classnames[angle_a]], 
                    kp_comp[id_classnames[angle_b]], 
                    kp_comp[id_classnames[angle_c]])
        )
        angles_coor_.append(
            (kp_comp[id_classnames[angle_a]], 
                kp_comp[id_classnames[angle_b]], 
                kp_comp[id_classnames[angle_c]])
        )
        angles_name_.append(
            (angle_a, angle_b, angle_c)
        )
    return angles_, angles_draw_, angles_coor_, angles_name_, kp_angles

def keypoint_pair_angles(kp, dict_angles):
    angles_ = []
    angles_draw_ = []
    angles_coor_ = []
    angles_name_ = []
    kp_angles = [(get_angle(kp[id_classnames[angle_a]], 
                    kp[id_classnames[angle_b]], 
                    kp[id_classnames[angle_c]]),
                get_angle_draw(kp[id_classnames[angle_a]], 
                    kp[id_classnames[angle_b]], 
                    kp[id_classnames[angle_c]]),
                (kp[id_classnames[angle_a]], 
                kp[id_classnames[angle_b]], 
                kp[id_classnames[angle_c]]),
                (angle_a, angle_b, angle_c))
                for angle_a, angle_b, angle_c in dict_angles]
    for angle_a, angle_b, angle_c in dict_angles:
        angles_.append((get_angle(kp[id_classnames[angle_a]], 
                    kp[id_classnames[angle_b]], 
                    kp[id_classnames[angle_c]])))
        angles_draw_.append(
            get_angle_draw(kp[id_classnames[angle_a]], 
                    kp[id_classnames[angle_b]], 
                    kp[id_classnames[angle_c]])
        )
        angles_coor_.append(
            (kp[id_classnames[angle_a]], 
            kp[id_classnames[angle_b]], 
            kp[id_classnames[angle_c]])
        )
        angles_name_.append(
            (angle_a, angle_b, angle_c)
        )
    return angles_, angles_draw_, angles_coor_, angles_name_, kp_angles

def show_result_angles_orin(img,
                            kpts,
                            skeleton=None,
                            classnames=None,
                            kpt_score_thr=0.0,
                            bbox_color='green',
                            pose_kpt_color=None,
                            pose_limb_color=None,
                            radius=4,
                            text_color=(255, 0, 0),
                            thickness=1,
                            font_scale=0.5,
                            win_name='',
                            angles_list=None,
                            show=False,
                            wait_time=0,
                            out_file=None):

    img_h, img_w, _ = img.shape
        # draw each point on image
    if pose_kpt_color is not None:
        kpts = kpts[:17]
        # print ("len(kpts)", len(kpts))
        # print ("len(pose_kpt_color)", len(pose_kpt_color))
        assert len(pose_kpt_color) == len(kpts)
        for kid, kpt in enumerate(kpts):
            if kid in [0, 1, 2, 3, 4]:
                continue
            x_coord, y_coord = int(kpt[0]), int(
                kpt[1])
            img_copy = img.copy()
            r, g, b = pose_kpt_color[kid]
            cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                        radius, (int(r), int(g), int(b)), -1)
            if classnames:
                cv2.putText(img_copy, "{}".format(classnames[kid]),(int(x_coord), int(y_coord)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (int(r), int(g), int(b)), 1)
            transparency = max(0, min(1, 0.5))
            cv2.addWeighted(
                img_copy,
                transparency,
                img,
                1 - transparency,
                0,
                dst=img)

    # draw limbs
    if skeleton is not None and pose_limb_color is not None:
        assert len(pose_limb_color) == len(skeleton)
        for sk_id, sk in enumerate(skeleton):
            if any(i in [1, 2, 3, 4, 5]for i in sk):
                continue
            pos1 = (int(kpts[sk[0] - 1][0]), int(kpts[sk[0] - 1][1]))
            pos2 = (int(kpts[sk[1] - 1][0]), int(kpts[sk[1] - 1][1]))
            middle12 = (int((pos1[0] + pos2[0])/2), int((pos1[1] + pos2[1])/2))
            if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                    and pos1[1] < img_h and pos2[0] > 0
                    and pos2[0] < img_w and pos2[1] > 0
                    and pos2[1] < img_h):
                img_copy = img.copy()
                X = (pos1[0], pos2[0])
                Y = (pos1[1], pos2[1])
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                angle = math.degrees(
                    math.atan2(Y[0] - Y[1], X[0] - X[1]))
                stickwidth = 2
                polygon = cv2.ellipse2Poly(
                    (int(mX), int(mY)),
                    (int(length / 2), int(stickwidth)), int(angle),
                    0, 360, 1)

                r, g, b = pose_limb_color[sk_id]
                cv2.fillConvexPoly(img_copy, polygon,
                                    (int(r), int(g), int(b)))
                transparency = max(
                    0,
                    min(1, 0.5))
                cv2.addWeighted(
                    img_copy,
                    transparency,
                    img,
                    1 - transparency,
                    0,
                    dst=img)

    for angle, sub_angle, list_coor, (na, nb, nc) in angles_list:
        r, g, b = pose_limb_color[id_classnames[nb]]
        all_x = 0
        all_y = 0
        list_coor = np.array(list_coor)
        plot_coor = np.average(list_coor, axis=0, weights=[0.1, 0.8, 0.1])

        from PIL import Image, ImageDraw
        #convert image opened opencv to PIL
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        #draw angle
        sub1_LKnee_Angle, sub2_LKnee_Angle = sub_angle
        ca, cb, cc = list_coor
        shape_LKnee = [(cb[0] - 15, cb[1] - 15), (cb[0] + 15, cb[1] + 15)]
        draw.arc(shape_LKnee, start=sub2_LKnee_Angle, end=sub1_LKnee_Angle, fill=(r, g, b))
        img = np.array(img_pil)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.putText(img, "{}".format(str(round(angle, 2))), (int(plot_coor[0]), int(plot_coor[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    idx_left_x = img_w - int(0.3*img_w)
    idx_left_h = img_h - int(0.4*img_h)
    left_angles = angles_list[int(len(angles_list)//2):]
    left_angles_name = dict_name_angles[int(len(angles_list)//2):]
    for (angle, _, _, (na, nb, nc)), ang_name in zip(left_angles, left_angles_name):
        r, g, b = pose_kpt_color[id_classnames[nb]]
        cv2.putText(img, "{}: {}".format(str(ang_name), str(round(angle, 2))), (idx_left_x, idx_left_h),
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, (int(r), int(g), int(b)), 1)
        idx_left_h = idx_left_h + 15

    idx_right_x = 9
    idx_right_h = img_h - int(0.4*img_h)
    right_angles = angles_list[:int(len(angles_list)//2)]
    right_angles_name = dict_name_angles[:int(len(angles_list)//2)]
    for (angle, _, _, (na, nb, nc)), ang_name in zip(right_angles, right_angles_name):
        r, g, b = pose_kpt_color[id_classnames[nb]]
        cv2.putText(img, "{}: {}".format(str(ang_name), str(round(angle, 2))), (idx_right_x, idx_right_h),
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, (int(r), int(g), int(b)), 1)
        idx_right_h = idx_right_h + 15

    if show:
        imshow(img, win_name, wait_time)

    if out_file is not None:
        imwrite(img, out_file)

    return img

def show_result_angles(img,
                        kpts,
                        skeleton=None,
                        classnames=None,
                        kpt_score_thr=0.0,
                        bbox_color='green',
                        pose_kpt_color_bool=None,
                        pose_kpt_color=None,
                        pose_limb_color=None,
                        radius=4,
                        text_color=(255, 0, 0),
                        thickness=1,
                        font_scale=0.5,
                        win_name='',
                        diff_angle_neg_threshold=None,
                        diff_angle=None,
                        angles_list=None,
                        show=False,
                        wait_time=0,
                        out_file=None):

    img_h, img_w, _ = img.shape

    # draw each point on image
    if pose_kpt_color is not None:
        kpts = kpts[:17]
        # print ("len(kpts)", len(kpts))
        # print ("len(pose_kpt_color)", len(pose_kpt_color))
        assert len(pose_kpt_color) == len(kpts)
        for kid, kpt in enumerate(kpts):
            if kid in [0, 1, 2, 3, 4]:
                continue
            x_coord, y_coord = int(kpt[0]), int(
                kpt[1])
            img_copy = img.copy()
            r, g, b = pose_kpt_color[kid]
            cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                        radius, (int(r), int(g), int(b)), -1)
            if pose_kpt_color_bool[kid]:
                # print ("HERER")
                cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                        30, (0, 0, 255), 2)
            # if classnames:
            #     cv2.putText(img_copy, "{}".format(classnames[kid]),(int(x_coord), int(y_coord)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (int(r), int(g), int(b)), 1)
            transparency = max(0, 0.5)
            cv2.addWeighted(
                img_copy,
                transparency,
                img,
                1 - transparency,
                0,
                dst=img)
    # draw limbs
    if skeleton is not None and pose_limb_color is not None:
        assert len(pose_limb_color) == len(skeleton)
        for sk_id, sk in enumerate(skeleton):
            if any(i in [1, 2, 3, 4, 5] for i in sk):
                continue
            pos1 = (int(kpts[sk[0] - 1][0]), int(kpts[sk[0] - 1][1]))
            pos2 = (int(kpts[sk[1] - 1][0]), int(kpts[sk[1] - 1][1]))
            middle12 = (int((pos1[0] + pos2[0])/2), int((pos1[1] + pos2[1])/2))
            if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                    and pos1[1] < img_h and pos2[0] > 0
                    and pos2[0] < img_w and pos2[1] > 0
                    and pos2[1] < img_h):
                img_copy = img.copy()
                X = (pos1[0], pos2[0])
                Y = (pos1[1], pos2[1])
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                angle = math.degrees(
                    math.atan2(Y[0] - Y[1], X[0] - X[1]))
                stickwidth = 2
                polygon = cv2.ellipse2Poly(
                    (int(mX), int(mY)),
                    (int(length / 2), int(stickwidth)), int(angle),
                    0, 360, 1)

                r, g, b = pose_limb_color[sk_id]
                cv2.fillConvexPoly(img_copy, polygon,
                                    (int(r), int(g), int(b)))
                cv2.addWeighted(
                    img_copy,
                    1,
                    img,
                    0,
                    0,
                    dst=img)
    for idx, (angle, sub_angle, list_coor, (na, nb, nc)) in enumerate(angles_list):
        r, g, b = pose_limb_color[id_classnames[nb]]
        all_x = 0
        all_y = 0
        list_coor = np.array(list_coor)
        plot_coor = np.average(list_coor, axis=0, weights=[0.1, 0.8, 0.1])

        from PIL import Image, ImageDraw
        #convert image opened opencv to PIL
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        #draw angle
        sub1_LKnee_Angle, sub2_LKnee_Angle = sub_angle
        ca, cb, cc = list_coor
        shape_LKnee = [(cb[0] - 15, cb[1] - 15), (cb[0] + 15, cb[1] + 15)]
        draw.arc(shape_LKnee, start=sub2_LKnee_Angle, end=sub1_LKnee_Angle, fill=(int(r), int(g), int(b)))
        img = np.array(img_pil)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if diff_angle is not None:
            cv2.putText(img, "{}".format(str(round(diff_angle[idx], 2))), (int(plot_coor[0]), int(plot_coor[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (int(r), int(g), int(b)), 1)
        else:
            cv2.putText(img, "{}".format(str(round(angle, 2))), (int(plot_coor[0]), int(plot_coor[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (int(r), int(g), int(b)), 1)

    # idx_left_x = img_w - int(0.3*img_w)
    # idx_left_h = img_h - int(0.4*img_h)
    # left_angles = angles_list[int(len(angles_list)//2):]
    # left_angles_name = dict_name_angles[int(len(angles_list)//2):]
    # for (angle, _, _, (na, nb, nc)), ang_name in zip(left_angles, left_angles_name):
    #     r, g, b = pose_kpt_color[id_classnames[nb]]
    #     cv2.putText(img, "{}: {}".format(str(ang_name), str(round(angle, 2))), (idx_left_x, idx_left_h),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (int(r), int(g), int(b)), 2)
    #     idx_left_h = idx_left_h + 15

    # idx_right_x = 9
    # idx_right_h = img_h - int(0.4*img_h)
    # right_angles = angles_list[:int(len(angles_list)//2)]
    # right_angles_name = dict_name_angles[:int(len(angles_list)//2)]
    # for (angle, _, _, (na, nb, nc)), ang_name in zip(right_angles, right_angles_name):
    #     r, g, b = pose_kpt_color[id_classnames[nb]]
    #     cv2.putText(img, "{}: {}".format(str(ang_name), str(round(angle, 2))), (idx_right_x, idx_right_h),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (int(r), int(g), int(b)), 2)
    #     idx_right_h = idx_right_h + 15

    # if show:
    #     imshow(img, win_name, wait_time)

    # if out_file is not None:
    #     imwrite(img, out_file)

    idx_left_x = 10
    idx_left_h = 10

    idx_right_x = img_w - int(0.35*img_w)
    idx_right_h = 10
    for idx_angle, (angle_neg, angle_dict) in enumerate(zip(diff_angle_neg_threshold, dict_angles)):
        if angle_neg:
            r, g, b = 0, 0, 255
            kp1, kp2 = skeleton[idx_angle]
            name1, name2, name3 = angle_dict
            display_text = "{}: {}".format(dict_name_angles[idx_angle], \
                                            str(round(diff_angle[idx_angle], 2)))
            cv2.putText(img, "{}".format(display_text), (idx_left_x, idx_left_h),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (int(r), int(g), int(b)), 1)
            idx_left_h = idx_left_h + 15
        else:
            r, g, b = 255, 0, 0
            kp1, kp2 = skeleton[idx_angle]
            name1, name2, name3 = angle_dict
            display_text = "{}: {}".format(dict_name_angles[idx_angle], " OK")
            cv2.putText(img, "{}".format(display_text), (idx_right_x, idx_right_h),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (int(r), int(g), int(b)), 1)
            idx_right_h = idx_right_h + 15

    return img

def show_result_original(img,
                kpts,
                action_bool,
                skeleton=None,
                kpt_score_thr=0.3,
                bbox_color='green',
                pose_kpt_color=None,
                pose_limb_color=None,
                radius=4,
                text_color=(255, 0, 0),
                thickness=1,
                font_scale=0.5,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None):
    """Draw `result` over `img`.
    Args:
        img (str or Tensor): The image to be displayed.
        result (list[dict]): The results to draw over `img`
            (bbox_result, pose_result).
        kpt_score_thr (float, optional): Minimum score of keypoints
            to be shown. Default: 0.3.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
            If None, do not draw keypoints.
        pose_limb_color (np.array[Mx3]): Color of M limbs.
            If None, do not draw limbs.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
            Default: 0.
        out_file (str or None): The filename to write the image.
            Default: None.
    Returns:
        Tensor: Visualized img, only if not `show` or `out_file`.
    """
    img_h, img_w, _ = img.shape
    action_draw_points = np.linspace(20, img_w-20, 8)
    red = 255, 0, 0
    blue = 0, 0, 255
    y_coord = 30
    y_text_coord = 10
    y_text_coord_below = 50

    # for idx_action, idx_draw_point in enumerate(range(0, 7, 1)):
    #     r, g, b = red if action_bool[idx_draw_point + 1] else blue
    #     cv2.line(img, (int(action_draw_points[idx_draw_point]), y_coord), (int(action_draw_points[idx_draw_point+1]), y_coord), (int(r), int(g), int(b)), 2)

    for idx_action, (x_coord, action) in enumerate(zip(action_draw_points, action_bool)):
        r, g, b = red if action else blue
        if idx_action == 0 or idx_action == 7:
            cv2.circle(img, (int(x_coord), y_coord), 10, (int(r), int(g), int(b)), -1)
            cv2.putText(img, "{}".format(idx_action+1),(int(x_coord - 2), int(y_text_coord)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (int(r), int(g), int(b)), 1)
        if idx_action == 0:
            cv2.putText(img, "begin", (int(x_coord - 10), int(y_text_coord_below)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (int(r), int(g), int(b)), 1)
        if idx_action == 7:
            cv2.putText(img, "end", (int(x_coord - 7), int(y_text_coord_below)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (int(r), int(g), int(b)), 1)

    # draw each point on image
    if pose_kpt_color is not None:
        kpts = kpts[:17]
        assert len(pose_kpt_color) == len(kpts)
        for kid, kpt in enumerate(kpts):
            # print (kid, kpt)
            x_coord, y_coord, kpt_score = int(kpt[0]), int(
                kpt[1]), kpt[2]
            if kpt_score > kpt_score_thr:
                img_copy = img.copy()
                r, g, b = pose_kpt_color[kid]
                cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                            radius, (int(r), int(g), int(b)), -1)
                transparency = max(0, min(1, kpt_score))
                cv2.addWeighted(
                    img_copy,
                    transparency,
                    img,
                    1 - transparency,
                    0,
                    dst=img)

    # draw limbs
    if skeleton is not None and pose_limb_color is not None:
        # print ("len(pose_limb_color)", len(pose_limb_color))
        # print ("len(skeleton)", len(skeleton))
        assert len(pose_limb_color) == len(skeleton)
        for sk_id, sk in enumerate(skeleton):
            pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1,
                                                        1]))
            pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1,
                                                        1]))
            if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                    and pos1[1] < img_h and pos2[0] > 0
                    and pos2[0] < img_w and pos2[1] > 0
                    and pos2[1] < img_h
                    and kpts[sk[0] - 1, 2] > kpt_score_thr
                    and kpts[sk[1] - 1, 2] > kpt_score_thr):
                img_copy = img.copy()
                X = (pos1[0], pos2[0])
                Y = (pos1[1], pos2[1])
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                angle = math.degrees(
                    math.atan2(Y[0] - Y[1], X[0] - X[1]))
                stickwidth = 2
                polygon = cv2.ellipse2Poly(
                    (int(mX), int(mY)),
                    (int(length / 2), int(stickwidth)), int(angle),
                    0, 360, 1)

                r, g, b = pose_limb_color[sk_id]
                cv2.fillConvexPoly(img_copy, polygon,
                                    (int(r), int(g), int(b)))
                transparency = max(
                    0,
                    min(
                        1, 0.5 *
                        (kpts[sk[0] - 1, 2] + kpts[sk[1] - 1, 2])))
                cv2.addWeighted(
                    img_copy,
                    transparency,
                    img,
                    1 - transparency,
                    0,
                    dst=img)

    if show:
        imshow(img, win_name, wait_time)

    if out_file is not None:
        imwrite(img, out_file)

    return img

def features_vectorizer(features):
    np_kp_ = []
    np_kp_full_ = []
    np_kp_pairs_ = []
    np_angle_ = []
    np_angle_draw_ = []
    kp_angle_coor_ = []
    kp_angle_name_ = []
    kp_angle_raw_ = []
    kp_coco_angles_ = []
    not_include_ = [1, 2, 3, 4, 5]
    img_ = []
    img_vis_ = []

    for img, vis_img, keypoints, keypoints_full, pair_keypoints, angles, angles_draw, angles_coor, angles_name, kp_coco_angles in features:
        img_.append(img)
        img_vis_.append(vis_img)
        np_kp_.append(np.array(keypoints))
        np_kp_full_.append(np.array(keypoints_full))
        np_kp_pairs_.append(pair_keypoints)
        kp_coco_angles_.append(kp_coco_angles)
        np_angle_.append(angles)
        np_angle_draw_.append(angles_draw)
        kp_angle_coor_.append(angles_coor)
        kp_angle_name_.append(angles_name)
        kp_angle_raw_.append(kp_coco_angles)

    np_kp_ = np.array(np_kp_)
    np_kp_full_ = np.array(np_kp_full_)
    np_kp_pairs_ = np.array(np_kp_pairs_)
    np_angle_ = np.array(np_angle_)
    np_angle_draw_ = np.array(np_angle_draw_)

    return img_, img_vis_, np_kp_, np_kp_full_, np_angle_, np_angle_draw_, kp_angle_coor_, kp_angle_name_, kp_angle_raw_, kp_coco_angles_, np_kp_pairs_


def vectorizer_kp(folder_path):
    np_kp_ = []
    raw_ = []
    image_file_ = []
    np_kp_pairs_ = []
    kp_name_pairs_ = []
    kp_angle_draw_ = []

    np_angle_ = []
    np_angle_draw_ = []
    kp_angle_coor_ = []
    kp_angle_name_ = []
    kp_angle_raw_ = []
    kp_coco_angles_ = []
    not_include_ = [1, 2, 3, 4, 5]

    for file_path_jpg in sorted(glob.glob(folder_path+"/*.jpg")):
        file_name_jpg = os.path.basename(file_path_jpg)
        file_name, jpg = file_name_jpg.split(".jpg")
        file_name_json = file_name + ".json"
        img = mmcv.imread(file_path_jpg)
        
        kp_pos = None
        with open(os.path.join(folder_path, file_name_json)) as f:
            anno = json.load(f)
            kp = anno['annotations'][0]['keypoints'][:17]
            kp_comp = np.array([[x, y] for (x, y, c) in kp])
            kp_name = anno['categories'][0]['keypoints'][:17]
            sk = anno['categories'][0]['skeleton']

            if len(kp) > 0:
                np_kp_.append(np.array(kp))
                raw_.append(anno)
                image_file_.append(file_path_jpg)
                kp_name_pairs_.append([(kp_name[p1-1], kp_name[p2-1]) for p1, p2 in sk \
                                    if p1 not in not_include and p2 not in not_include])

                np_kp_pairs_.append([np.array(kp_comp[p1-1] - kp_comp[p2-1]) for p1, p2 in sk \
                                    if p1 not in not_include and p2 not in not_include])
                angles, angles_draw, angles_coor, angles_name, kp_coco_angles = pair_angles(anno, dict_angles)
                kp_coco_angles_.append(kp_coco_angles)
                np_angle_.append(angles)
                np_angle_draw_.append(angles_draw)
                kp_angle_coor_.append(angles_coor)
                kp_angle_name_.append(angles_name)
                kp_angle_raw_.append(kp_coco_angles)


    np_kp_ = np.array(np_kp_)
    np_kp_pairs_ = np.array(np_kp_pairs_)
    np_angle_ = np.array(np_angle_)
    np_angle_draw_ = np.array(np_angle_draw_)

    return np_kp_, raw_, image_file_,  np_angle_, np_angle_draw_, \
            kp_angle_coor_, kp_angle_name_, kp_angle_raw_, kp_coco_angles_, np_kp_pairs_

def split_in_face_torso(inkp):
    input_face = inkp[0:5]
    input_torso = inkp[5:17]
    return input_face, input_torso

def transform(kp_model, kp_person): # kp_model (X,Y,1.0)
    A, res, rank, s = np.linalg.lstsq(kp_model, kp_person)
    A[np.abs(A) < 1e-10] = 0
    ret = np.dot(kp_model, A)
    # c, R, t = umeyama(kp_model, kp_person)
    # ret = kp_model.dot(c*R) + t
    return ret

def pair_new_angles(kp, dict_angles):
    kp_comp = kp[:, :-1]
    angles_ = []
    angles_draw_ = []
    angles_coor_ = []
    angles_name_ = []
    kp_angles = [(get_angle(kp_comp[id_classnames[angle_a]], 
                    kp_comp[id_classnames[angle_b]], 
                    kp_comp[id_classnames[angle_c]]),
                get_angle_draw(kp_comp[id_classnames[angle_a]], 
                    kp_comp[id_classnames[angle_b]], 
                    kp_comp[id_classnames[angle_c]]),
                (kp_comp[id_classnames[angle_a]], 
                kp_comp[id_classnames[angle_b]], 
                kp_comp[id_classnames[angle_c]]),
                (angle_a, angle_b, angle_c))
                for angle_a, angle_b, angle_c in dict_angles]
    for angle_a, angle_b, angle_c in dict_angles:
        angles_.append((get_angle(kp_comp[id_classnames[angle_a]], 
                    kp_comp[id_classnames[angle_b]], 
                    kp_comp[id_classnames[angle_c]])))
        angles_draw_.append(
            get_angle_draw(kp_comp[id_classnames[angle_a]], 
                    kp_comp[id_classnames[angle_b]], 
                    kp_comp[id_classnames[angle_c]])
        )
        angles_coor_.append(
            (kp_comp[id_classnames[angle_a]], 
                kp_comp[id_classnames[angle_b]], 
                kp_comp[id_classnames[angle_c]])
        )
        angles_name_.append(
            (angle_a, angle_b, angle_c)
        )
    return angles_, angles_draw_, angles_coor_, angles_name_, kp_angles

def draw_pose(img, draw_pose_bool, display_circle, kpts, angles, display_id, display_angles, skeleton, pose_kpt_color, pose_limb_color):
    img_h, img_w, _ = img.shape
    # draw keypoints
    img = img.copy()
    for kid, kpt in enumerate(kpts):
        if kid in [0, 1, 2, 3, 4]:
            continue
        if kid in display_id:
            img_copy = img.copy()
            x_coord, y_coord, kpt_score = kpt
            x_coord, y_coord = int(x_coord), int(y_coord)
            r, g, b = pose_kpt_color[kid]
            cv2.circle(img, (int(x_coord), int(y_coord)),
                                4, (int(r), int(g), int(b)), -1)
            # transparency = max(0, min(1, kpt_score))
            transparency = 0.2
            cv2.addWeighted(img_copy,
                            transparency,
                            img,
                            1 - transparency,
                            0,
                            dst=img)

    # draw limbs
    list_kp_id_pose = []
    for idx, (a, b, c) in enumerate(dict_angles):
        if display_angles[idx]:
            list_kp_id_pose.append([id_classnames[a], id_classnames[b], id_classnames[c]])
    # print ("list_kp_id_pose", list_kp_id_pose)

    if skeleton is not None and pose_limb_color is not None:
        assert len(pose_limb_color) == len(skeleton)
        for sk_id, sk in enumerate(skeleton):
            sk_1, sk_2 = sk
            sk_1 = sk_1 - 1
            sk_2 = sk_2 - 1
            for kp_id_pose in list_kp_id_pose:
                if all(i in kp_id_pose for i in [sk_1, sk_2]):
                    kpt_x_1, kpt_y_1, _ = kpts[sk_1]
                    kpt_x_2, kpt_y_2, _ = kpts[sk_2]
                    pos1 = (int(kpt_x_1), int(kpt_y_1))
                    pos1_x, pos1_y = pos1
                    pos2 = (int(kpt_x_2), int(kpt_y_2))
                    pos2_x, pos2_y = pos2
                    middle12 = (int((pos1_x + pos2_x)/2), int((pos1_y + pos2_y)/2))
                    middle12_x, middle12_y = middle12
                    if (kpt_x_1 > 0 and kpt_x_1 < img_w 
                        and kpt_y_1 > 0 and kpt_y_1 < img_h 
                        and kpt_x_2 > 0 and kpt_x_2 < img_w 
                        and kpt_y_2 > 0 and kpt_y_2 < img_h):
                        img_copy = img.copy()
                        X = (pos1_x, pos2_x)
                        Y = (pos1_y, pos2_y)
                        mX = np.mean(X)
                        mY = np.mean(Y)
                        length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                        angle = math.degrees(
                            math.atan2(Y[0] - Y[1], X[0] - X[1]))
                        stickwidth = 2
                        polygon = cv2.ellipse2Poly(
                            (int(mX), int(mY)),
                            (int(length / 2), int(stickwidth)), int(angle),
                            0, 360, 1)
                        r, g, b = pose_limb_color[sk_id]
                        cv2.fillConvexPoly(img_copy, polygon, (int(r), int(g), int(b)), 4)
                        transparency = 0.5
                        cv2.addWeighted(
                            img_copy,
                            transparency,
                            img,
                            1 - transparency,
                            0,
                            dst=img)
    
    for idx, ((angle, sub_angle, list_coor, (na, nb, nc)), display) in enumerate(zip(angles, display_angles)):
        if display:
            r, g, b = pose_kpt_color[id_classnames[nb]]
            list_coor = np.array(list_coor)
            plot_coor = np.average(list_coor, axis=0, weights=[0.1, 0.8, 0.1])

            #convert image opened opencv to PIL
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)

            #draw angle
            sub1_LKnee_Angle, sub2_LKnee_Angle = sub_angle
            ca, cb, cc = list_coor
            x_coord, y_coord = cb
            radius_rand = random.randint(5, 15)
            shape_LKnee = [(cb[0] - radius_rand, cb[1] - radius_rand), (cb[0] + radius_rand, cb[1] + radius_rand)]
            draw.arc(shape_LKnee, start=sub2_LKnee_Angle, end=sub1_LKnee_Angle, fill=(int(r), int(g), int(b)))
            img = np.array(img_pil)
            if draw_pose_bool[idx] and display_circle:
                cv2.circle(img, (int(x_coord), int(y_coord)),
                                        30, (0, 0, 255), 2)

    return img

def draw_transform(img, pose_bool_list, display_circle_list, kp_list, angle_list, display_id_list, bool_list, skeleton, pose_kpt_color_list, pose_limb_color_list):
    img = img.copy()
    for kp, pose_bool, display_circle, al, display_id, b, kp_color, limb_color in zip(kp_list, pose_bool_list, display_circle_list, angle_list, display_id_list, bool_list, pose_kpt_color_list, pose_limb_color_list):
        img = draw_pose(img, pose_bool, display_circle, kp, al, display_id, b, skeleton, kp_color, limb_color)

    return img

def return_text(part, angle, flag):
    if flag == 1:
        if part == "RKnee":
            return "Put your Rleg straight " +str(angle) + " angle"
        if part == "LKnee":
            return "Put your Lleg straight "+ str(angle) + " angle"
        if part == "RHip" or part == "LHip":
            return "Put your Hip straight " + str(angle) + " angle"
        if part == "RElbow":
            return "Put your Rarm straight " + str(angle) + " angle"
        if part == "LElbow":
            return "Put your Larm straight " + str(angle) + " angle"
        if part == "LShoulder":
            return "Put your LShoulder Wider " + str(angle) + " angle"
        if part == "LShoulderElbow":
            return "Put your LShoulderElbow Wider " + str(angle) + " angle"
        if part == "RShoulder":
            return "Put your RShoulder Wider " + str(angle) + " angle"
        if part == "RShoulderElbow":
            return "Put your RShoulderElbow Wider " + str(angle) + " angle"
        return "Put your " + part + " straight" + str(angle) + " angle"
    if flag == 0:
        if part == "RKnee":
            return "Put your Rleg bent " + str(angle) + " angle"
        if part == "LKnee":
            return "Put your Lleg bent " + str(angle) + " angle"
        if part == "RHip" or part == "LHip":
            return "Put your Hip bent " + str(angle) + " angle"
        if part == "RElbow":
            return "Put your Rarm bent " + str(angle) + " angle"
        if part == "LElbow":
            return "Put your Larm bent " + str(angle) + " angle"
        if part == "LShoulder":
            return "Put your LShoulder narrow " + str(angle) + " angle"
        if part == "LShoulderElbow":
            return "Put your LShoulderElbow narrow " + str(angle) + " angle"
        if part == "RShoulder":
            return "Put your RShoulder narrow " + str(angle) + " angle"
        if part == "RShoulderElbow":
            return "Put your RShoulderElbow narrow " + str(angle) + " angle"
        return "Put your " + part + " bent" + str(angle) + " angle"

    return ""

def put_instruction(img, bool_display, angle_transform, angle_neg, pair_angles):
    img_h, img_w, _ = img.shape
    idx_x = 2
    idx_h = 10
    display_len = [i for i in range(0, len(pair_angles)) if bool_display[i]]
    if len(display_len) > 0:
        cv2.rectangle(img, (2, 0), (310, 12*len(display_len)), (0, 255, 0), cv2.FILLED)
    for (idx_angle, (angle_num, a_tran, a_neg)) in enumerate(zip(pair_angles, angle_transform, angle_neg)):
        text_display  = ""
        if angle_num > 0:
            if a_tran >= 180 and a_neg >= 180:
                text_display = return_text(dict_name_angles[idx_angle], str(round(angle_num, 2)), 0)
            if a_tran >= 180 and a_neg < 175:
                text_display = return_text(dict_name_angles[idx_angle], str(round(angle_num, 2)), 1)
            if a_tran >= 180 and a_neg >= 175 and a_neg < 180:
                text_display = return_text(dict_name_angles[idx_angle], str(round(angle_num, 2)), 0)
            if a_tran < 180 and a_neg < 180:
                text_display = return_text(dict_name_angles[idx_angle], str(round(angle_num, 2)), 1)
            if a_tran >= 180 and a_neg < 180:
                text_display = return_text(dict_name_angles[idx_angle], str(round(angle_num, 2)), 1)
        else:
            if a_neg > 180:
                text_display = return_text(dict_name_angles[idx_angle], str(round(angle_num, 2)), 1)
            else:
                text_display = return_text(dict_name_angles[idx_angle], str(round(angle_num, 2)), 0)
        if bool_display[idx_angle]:
            r, g, b = 0, 0, 255
            cv2.putText(img, "{}".format(text_display), (idx_x, idx_h),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (int(r), int(g), int(b)), 1)
            idx_h = idx_h + 12

    return img

from PIL import Image, ImageDraw
import random 

def analyzeAngle(room):
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    class args:
        show = True
        det_config = 'mmdetection/configs/detr/detr_r50_8x4_150e_coco.py'
        det_config = os.path.join(folder_mmpose_path, det_config)
        det_checkpoint = 'http://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'
        pose_config = 'configs/top_down/hrnet/coco/hrnet_w48_coco_256x192.py'
        pose_config = os.path.join(folder_mmpose_path, pose_config)
        pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
        out_video_root = 'vis_results_{}'.format(time.time())
        video_path = ''
        device = 'cuda:0'
        bbox_thr = 0.3
        kpt_thr = 0.3

    assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    np_kp_pos, raw_pos, image_file_pos, np_angle_pos, \
        np_angle_draw_pos, kp_angle_coor_pos, kp_angle_name_pos, \
        kp_angle_raw_pos, kp_coco_angles_pos, np_kp_pairs_pos = vectorizer_kp(
            folder_pos)

    det_model = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device)

    dataset = pose_model.cfg.data['test']['type']
    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True


    # optional
    idx_video = 1
    return_heatmap = False
    write_video = False
    action_bool = [False]*8

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    start_time = time.time()

    predict_queue = []
    full_queue = []
    temp_queue = []
    for msg in img_sub.listen():
        mess = msg['data']
        # print("mess: ", mess)
        if mess != 1:
            images = []
            annotations = []

            mess_decoded = mess.decode('UTF-8')
            img_path = mess_decoded
            img_original = cv2.imread(img_path)
            img = img_original.copy()

            time_s = time.time()

            # test a single image, the resulting box is (x1, y1, x2, y2)
            mmdet_results = inference_detector(det_model, img)

            # keep the person class bounding boxes.
            person_bboxes = process_mmdet_results(mmdet_results)

            # test a single image, with a list of bboxes.
            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                img,
                person_bboxes,
                bbox_thr=args.bbox_thr,
                format='xyxy',
                dataset=dataset,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)
            print (1./(time.time() - time_s))

            # save image and keypoints
            bbox = []
            pose = []
            for res in pose_results:
                bbox.extend(res['bbox'])
                pose.extend(res['keypoints'])
            
            # show the results
            keypoints = []
            keypoints_full = []
            for po in pose:
                x, y, c = po
                keypoints.append([int(x), int(y)])
                keypoints_full.append([int(x), int(y), int(c)])
            keypoints = np.array(keypoints)
            keypoints_full = np.array(keypoints_full)
        
            bboxes = []
            for bb in bbox:
                x, y, w, h, c = bb
                bboxes.append([int(x), int(y), int(w), int(h), 1.0])

            if len(bboxes) > 0:
                pair_keypoints = np.array([np.array(keypoints[p1-1] - keypoints[p2-1]) for p1, p2 in skeleton if p1 not in not_include and p2 not in not_include])
                comp_keypoints = pair_keypoints[None, :, :] # 1 x 12 x 2
                p12 = pairwise_angles(comp_keypoints, np_kp_pairs_pos)
                p12_sum = np.sum(p12, axis=-1)
                p12_argmin = np.argmin(p12_sum, axis=-1)
                
                predict_queue.append(p12_argmin)
                c = collections.Counter(predict_queue)        
                m_queue = c.most_common()

                if len(m_queue) > 0:
                    # print (m_queue)
                    # angles, angles_draw, angles_coor, angles_name, kp_coco_angles = keypoint_pair_angles(keypoints, dict_angles)
                    for m in m_queue:
                        mk, mv = m
                        # print ("mk: {} mv: {}".format(mk, mv))

                        if mk == 0 and mv >= 15:
                            action_bool[mk] = True
                            write_video = True
                            vis_img = show_result_original(img,
                                            np.array(pose),
                                            action_bool,
                                            skeleton,
                                            pose_kpt_color=pose_kpt_color,
                                            pose_limb_color=pose_limb_color,
                                            show=False)
                            image_queue = full_queue[-20:]
                        elif mk == 7 and mv >= 4 and write_video:
                            action_bool[mk] = True
                            write_video = False                            
                            pose_result_folder = time.time()

                            img_neg, img_vis_neg, np_kp_neg, np_kp_full_neg, np_angle_neg, np_angle_draw_neg, kp_angle_coor_neg, \
                                kp_angle_name_neg, kp_angle_raw_neg, kp_coco_angles_neg, np_kp_pairs_neg  = features_vectorizer(image_queue)

                            return_json = []
                            for idx_pos, (np_kp_, pairs_, image_file_, raw_) in enumerate(zip(np_kp_pos, np_kp_pairs_pos, image_file_pos, raw_pos)): 
                                time_start = time.time()
                                pairs_temp = pairs_[None, :, :]

                                p12 = pairwise_angles(pairs_temp, np_kp_pairs_neg)
                                p12_sum = np.sum(p12, axis=-1)
                                p12_argmin = np.argmin(p12_sum, axis=-1)

                                p12_diff = np_angle_pos[idx_pos] - np_angle_neg[p12_argmin]
                                p12_abs = np.abs(p12_diff)
                                p12_threshold = p12_abs > threshold

                                kp_angle_raw_pos_temp_ = kp_coco_angles_pos[idx_pos]
                                kp_angle_raw_neg_temp_ = kp_coco_angles_neg[p12_argmin]

                                diff_angle_pos = p12_diff
                                diff_angle_pos_bool = p12_threshold

                                image_pos_temp = mmcv.imread(image_file_)
                                image_neg_temp = img_neg[p12_argmin].copy()

                                color_kpt = np.array([
                                    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18
                                ])

                                color_limb = np.array([
                                    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18
                                ])

                                color_kpt_bool = np.array([
                                    False, False, False, False, False, False, False, False,  
                                    False, False, False, False, False, False, False, False, False
                                ])

                                pos_face, pos_torso = split_in_face_torso(np_kp_)
                                neg_face, neg_torso = split_in_face_torso(np_kp_full_neg[p12_argmin])
                                # print ("np_kp_: ", np_kp_.shape)
                                # print ("np_kp_full_neg: ", np_kp_full_neg[p12_argmin].shape)
                                temp_pos_transform_face = transform(pos_face, neg_face)
                                temp_pos_transform_torso = transform(pos_torso, neg_torso)

                                temp_pos_transform = np.vstack((temp_pos_transform_face, temp_pos_transform_torso))

                                angles_transform, angles_draw_transform, angles_coor_transform, \
                                        angles_name_transform, kp_coco_angles_transform = pair_new_angles(temp_pos_transform, dict_angles)

                                pair_angles_ = angles_transform - np_angle_neg[p12_argmin]
                                display_angles_transform = np.abs(pair_angles_) >= 20
                                # print (display_angles_transform)

                                display_kp_id_transform = []
                                for idx, (a, b, c) in enumerate(dict_angles):
                                    if display_angles_transform[idx]:
                                        display_kp_id_transform.extend([id_classnames[a], id_classnames[b], id_classnames[c]])
                                # print ("display_kp_id_transform", display_kp_id_transform)

                                display_angles = [True, True, True, True, True, True, True, True, True, True, True, True]
                                display_kp_id = []
                                for idx, (a, b, c) in enumerate(dict_angles):
                                    if display_angles[idx]:
                                        display_kp_id.extend([id_classnames[a], id_classnames[b], id_classnames[c]])
                                # print ("display_kp_id", display_kp_id)

                                image_neg_temp_copy = image_neg_temp.copy()
                                img_draw_temp = draw_transform(image_neg_temp_copy, \
                                                        [display_angles_transform, display_angles_transform], \
                                                        [True, False], \
                                                        [np_kp_full_neg[p12_argmin], temp_pos_transform], \
                                                        [kp_angle_raw_neg_temp_, kp_coco_angles_transform], \
                                                        [display_kp_id, display_kp_id_transform], \
                                                        [display_angles, display_angles_transform], \
                                                        skeleton, \
                                                        [pose_kpt_color, pose_kpt_color_correct], \
                                                        [pose_limb_color, pose_limb_color_correct])

                                # img_draw_temp = put_instruction(img_draw_temp, display_angles_transform, angles_transform, np_angle_neg[p12_argmin], pair_angles_)

                                img, bool_display, angle_transform, angle_neg, pair_angles = img_draw_temp, display_angles_transform, angles_transform, np_angle_neg[p12_argmin], pair_angles_
                                pose_json = {}
                                pose_json['name'] = "{}.png".format(idx_pos)
                                pose_json['pose'] = []
                                for (idx_angle, (angle_num, a_tran, a_neg)) in enumerate(zip(pair_angles, angle_transform, angle_neg)):
                                    temp_angle = {}
                                    temp_angle['name'] = dict_name_angles[idx_angle]
                                    temp_angle['angle'] = str(round(angle_num, 2))
                                    pose_json['pose'].append(temp_angle)

                                image_pos_temp = show_result_angles(image_pos_temp,
                                                    np_kp_[:17],
                                                    original_skeleton,
                                                    classnames,
                                                    diff_angle_neg_threshold=diff_angle_pos_bool,
                                                    diff_angle=diff_angle_pos,
                                                    angles_list=kp_angle_raw_pos_temp_,
                                                    pose_kpt_color_bool=color_kpt_bool,
                                                    pose_kpt_color=palette[color_kpt],
                                                    pose_limb_color=palette[color_limb])

                                img_draw = img_draw_temp

                                if img_draw is not None:
                                    img_draw_name = os.path.basename(image_file_)
                                    os.makedirs(os.path.join('static/result/', args.out_video_root, str(pose_result_folder)), exist_ok=True)
                                    img_draw_path = os.path.join(args.out_video_root, str(pose_result_folder), 
                                                        f'vis_{current_timestamp}_{img_draw_name}.png')
                                    
                                    img_draw_pos_path = os.path.join(args.out_video_root, str(pose_result_folder), 
                                                        f'vis_{current_timestamp}_{img_draw_name}_.png')
                                    
                                    mmcv.imwrite(img_draw, os.path.join('static/result/', img_draw_path))
                                    mmcv.imwrite(image_pos_temp, os.path.join('static/result/', img_draw_pos_path))
                                    pose_json['path'] = img_draw_path
                                return_json.append(pose_json)

                            # from pprint import pprint 
                            # pprint (return_json)
                            print (return_json)
                            socket.emit('analyzeAngle', return_json, room=room)

                            idx_video = idx_video + 1
                            vis_img = show_result_original(img,
                                            np.array(pose[:17]),
                                            action_bool,
                                            skeleton,
                                            pose_kpt_color=pose_kpt_color,
                                            pose_limb_color=pose_limb_color,
                                            show=False)
                            action_bool = [False]*8

                            image_queue = []
                        else:
                            vis_img = show_result_original(img,
                                            np.array(pose[:17]),
                                            action_bool,
                                            skeleton,
                                            pose_kpt_color=pose_kpt_color,
                                            pose_limb_color=pose_limb_color,
                                            show=False)

                if len(predict_queue) > 20:
                    predict_queue.pop(0)

                angles, angles_draw, angles_coor, angles_name, kp_coco_angles = keypoint_pair_angles(keypoints, dict_angles)
                # vis_img = img
                full_queue.append((img_original, vis_img, keypoints, keypoints_full, pair_keypoints, angles, angles_draw, angles_coor, angles_name, kp_coco_angles))
                vis_img = show_result_original(img,
                        np.array(pose[:17]),
                        action_bool,
                        skeleton,
                        pose_kpt_color=pose_kpt_color,
                        pose_limb_color=pose_limb_color,
                        show=False)

                if save_out_video and write_video:
                    image_queue.append((img_original, vis_img, keypoints, keypoints_full, pair_keypoints, angles, angles_draw, angles_coor, angles_name, kp_coco_angles))

                if write_video == False:
                    action_bool = [False]*8




