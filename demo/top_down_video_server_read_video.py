import os
import time
import json
from argparse import ArgumentParser
import collections

import math
from numpy import linalg as LA
import mmcv
import cv2
import datetime
import numpy as np
import math
import glob
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_top_down_pose_model, init_pose_model, vis_pose_result)
import redis
import copy

# connect with redis server as Bob
queue_name = "img_live_demo"

img_redis = redis.Redis(host='localhost', port=6379, db=0)
img_sub = img_redis.pubsub()

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

skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
pose_limb_color = palette[[
            16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16
        ]]
pose_kpt_color = palette[[
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16
]]

FREQ_START_SPLIT = 20
FREQ_END_SPLIT = 3

threshold = 10
w, h = int(477), int(900)
size = (w*5, h)
not_include = [1, 2, 3, 4, 5]
# frontal view
current_timestamp = str(time.time())
folder_path = "./golf_pose"
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
        print ("len(kpts)", len(kpts))
        print ("len(pose_kpt_color)", len(pose_kpt_color))
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
        print ("len(kpts)", len(kpts))
        print ("len(pose_kpt_color)", len(pose_kpt_color))
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
                print ("HERER")
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
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # if diff_angle is not None:
        #     cv2.putText(img, "{}".format(str(round(diff_angle[idx], 2))), (int(plot_coor[0]), int(plot_coor[1])),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (int(r), int(g), int(b)), 1)
        # else:
        #     cv2.putText(img, "{}".format(str(round(angle, 2))), (int(plot_coor[0]), int(plot_coor[1])),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (int(r), int(g), int(b)), 1)

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
        assert len(pose_kpt_color) == len(kpts)
        for kid, kpt in enumerate(kpts):
            print (kid, kpt)
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
        assert len(pose_limb_color) == len(skeleton)
        for sk_id, sk in enumerate(skeleton):
            print (sk_id, sk)

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

    for img, vis_img, keypoints, pair_keypoints, angles, angles_draw, angles_coor, angles_name, kp_coco_angles in features:
        img_.append(img)
        img_vis_.append(vis_img)
        np_kp_.append(np.array(keypoints))
        np_kp_pairs_.append(pair_keypoints)
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

    return img_, img_vis_, np_kp_, np_angle_, np_angle_draw_, kp_angle_coor_, kp_angle_name_, kp_angle_raw_, kp_coco_angles_, np_kp_pairs_


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

def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    np_kp_pos, raw_pos, image_file_pos, np_angle_pos, \
        np_angle_draw_pos, kp_angle_coor_pos, kp_angle_name_pos, \
            kp_angle_raw_pos, kp_coco_angles_pos, np_kp_pairs_pos = vectorizer_kp(folder_pos)

    det_model = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, device=args.device)

    dataset = pose_model.cfg.data['test']['type']
    cap = cv2.VideoCapture(args.video_path)
    videoWriter = None
    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(os.path.join(args.out_video_root, f'vis_{os.path.basename(args.video_path)}'), fourcc, fps, size)

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
    while (cap.isOpened()):
        images = []
        annotations = []

        flag, img = cap.read()
        if not flag:
            break

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
        for po in pose:
            x, y, c = po
            keypoints.append([int(x), int(y)])
        keypoints = np.array(keypoints)
    
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
            print (p12_argmin)
            
            predict_queue.append(p12_argmin)
            c = collections.Counter(predict_queue)        
            m_queue = c.most_common()

            if len(m_queue) > 0:
                print (m_queue)
                # angles, angles_draw, angles_coor, angles_name, kp_coco_angles = keypoint_pair_angles(keypoints, dict_angles)

                for m in m_queue:
                    mk, mv = m
                    print ("mk: {} mv: {}".format(mk, mv))

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
                        w, h, c = img.shape
                        # size = (int(w), int(h))
                        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        # videoWriter = cv2.VideoWriter(
                        #                 os.path.join(args.out_video_root, 
                        #                             f'vis_{current_timestamp}_{idx_video}.mp4'), 
                        #                             fourcc, fps, size
                        # )
                        # for idx_im, (image_temp, keypoints, pair_keypoints, angles, angles_draw, angles_coor, angles_name, kp_coco_angles) in enumerate(image_queue):
                            # videoWriter.write(image_temp)
                            # cv2.imwrite(os.path.join(args.out_video_root,  f'vis_{current_timestamp}_{idx_video}_{idx_im}.png'), image_temp)
                        
                        img_neg, img_vis_neg, np_kp_neg, np_angle_neg, np_angle_draw_neg, kp_angle_coor_neg, \
                            kp_angle_name_neg, kp_angle_raw_neg, kp_coco_angles_neg, np_kp_pairs_neg  = features_vectorizer(image_queue)

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
                            image_neg_temp = img_neg[p12_argmin]

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

                            for idx, (angle_diff , angle_bool) in enumerate(zip(diff_angle_pos, diff_angle_pos_bool)):
                                if angle_bool:
                                    list_angle_name = dict_angles[idx]
                                    print (list_angle_name)
                                    first, middle, last = list_angle_name
                                    id_class_kpt = id_classnames[middle]
                                    color_kpt_bool[id_class_kpt] = True

                            img_draw_temp = show_result_angles(image_neg_temp,
                                                np_kp_neg[p12_argmin],
                                                skeleton,
                                                classnames,
                                                diff_angle_neg_threshold=diff_angle_pos_bool,
                                                diff_angle=diff_angle_pos,
                                                angles_list=kp_angle_raw_neg_temp_,
                                                pose_kpt_color_bool=color_kpt_bool,
                                                pose_kpt_color=palette[color_kpt],
                                                pose_limb_color=palette[color_limb])

                            img_draw_temp = show_result_angles_orin(img_draw_temp,
                                            np_kp_neg[p12_argmin],
                                            skeleton,
                                            classnames,
                                            angles_list=kp_angle_raw_neg_temp_,
                                            pose_kpt_color=pose_kpt_color,
                                            pose_limb_color=pose_limb_color)

                            # image_pos_temp = image_pos_temp
                            image_pos_temp = show_result_angles_orin(image_pos_temp,
                                            np_kp_,
                                            skeleton,
                                            classnames,
                                            angles_list=kp_angle_raw_pos_temp_,
                                            pose_kpt_color=pose_kpt_color,
                                            pose_limb_color=pose_limb_color)

                            img_draw_neg = cv2.resize(img_draw_temp, (int(w), int(h)))
                            img_draw_pos = cv2.resize(image_pos_temp, (int(w*4), int(h)))

                            img_draw = np.hstack((img_draw_neg, img_draw_pos))

                            if img_draw is not None:
                                img_draw_name = os.path.basename(image_file_)
                                img_draw_path = os.path.join(args.out_video_root, 
                                                    f'vis_{current_timestamp}_{img_draw_name}_{time.time()}.png')

                                
                                mmcv.imwrite(img_draw, img_draw_path)

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
                        videoWriter.release()
                        cv2.destroyAllWindows()
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
            full_queue.append((img, vis_img, keypoints, pair_keypoints, angles, angles_draw, angles_coor, angles_name, kp_coco_angles))
            vis_img = show_result_original(img,
                    np.array(pose[:17]),
                    action_bool,
                    skeleton,
                    pose_kpt_color=pose_kpt_color,
                    pose_limb_color=pose_limb_color,
                    show=False)

            if args.show:
                # videoFullWriter.write(vis_img)
                cv2.imshow('Image', vis_img)

            if save_out_video and write_video:
                videoWriter.write(vis_img)
                image_queue.append((img, vis_img, keypoints, pair_keypoints, angles, angles_draw, angles_coor, angles_name, kp_coco_angles))

            if write_video == False:
                action_bool = [False]*8

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print ("extract pose: {}".format(time.time() - start_time))


    cap.release()
    if save_out_video:
        videoWriter.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

#python demo/top_down_video_demo_with_mmdet_8poses.py  mmdetection/configs/detr/detr_r50_8x4_150e_coco.py  http://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth     configs/top_down/hrnet/coco/hrnet_w48_coco_256x192.py     https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth     --video-path ./positive/2.mp4    --out-video-root vis_results$(date "+%Y.%m.%d-%H.%M.%S") --show
#python demo/top_down_video_demo_with_mmdet_8poses.py  mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py  http://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco-139f5633.pth     configs/top_down/mobilenet_v2/coco/mobilenetv2_coco_384x288.py     https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_coco_384x288-26be4816_20200727.pth     --video-path ./demo/3_4.mp4    --out-video-root vis_results$(date "+%Y.%m.%d-%H.%M.%S") --show
#python demo/top_down_video_demo_with_mmdet_8poses.py  mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py  http://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco-139f5633.pth     configs/top_down/hrnet/coco/hrnet_w48_coco_256x192.py     https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth     --video-path ./demo/3_4.mp4    --out-video-root vis_results$(date "+%Y.%m.%d-%H.%M.%S") --show
