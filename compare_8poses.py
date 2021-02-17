import json
import os
import numpy as np
import mmcv
from matplotlib import pyplot as plt
import cv2
import math
from pylab import rcParams
import copy
import time
from sklearn import metrics
import glob
import math
from numpy import linalg as LA
from PIL import Image, ImageDraw
import random

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

def split_in_face_legs_torso(inkp):
    input_face = inkp[0:5]
    input_torso = inkp[5:10]
    input_leg = inkp[10:17]
    return input_face, input_torso, input_leg

def split_in_face_torso(inkp):
    input_face = inkp[0:5]
    input_torso = inkp[5:17]
    return input_face, input_torso

id_classnames = {v: k for k, v in classnames.items()}

palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], 
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11]]
pose_limb_color = palette[[18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18]]
pose_limb_color_correct = palette[[16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]]
pose_kpt_color = palette[[18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18]]
pose_kpt_color_correct = palette[[16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]]

rcParams['figure.figsize'] = 10, 10
folder_path = "./golf_pose"
# frontal view
folder_pos = os.path.join(folder_path, "positive", "3_3")
# folder_neg = os.path.join(folder_path, "negative", "kakao_905") #women, kakao_905
folder_neg = os.path.join(folder_path, "negative", "women") #women, kakao_905

fps = 30
w, h = 480, 852
# w, h = 800, 1500
size = (w*2, h)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter("./out_{}.mp4".format(time.time()), fourcc, fps, size)

np_pos = []
raw_pos = []
image_file_pos = []
np_kp_pos_pairs = []
kp_name_pos_pairs = []

not_include = [1, 2, 3, 4, 5]

import numpy as np
import numpy.linalg

def umeyama(P, Q):
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(P, axis=0).sum()
    c = 1/varP * np.sum(S) # scale factor

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)

    return c, R, t

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

def pair_angles(kp, dict_angles):
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

pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))]) 
unpad = lambda x: x[:, :-1]

def transform(kp_model, kp_person): # kp_model (X,Y,1.0)
    A, res, rank, s = np.linalg.lstsq(kp_model, kp_person)
    A[np.abs(A) < 1e-10] = 0
    ret = np.dot(kp_model, A)
    # c, R, t = umeyama(kp_model, kp_person)
    # ret = kp_model.dot(c*R) + t
    return ret

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
    print ("list_kp_id_pose", list_kp_id_pose)

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

list_transform = []
for idx_pos, (pairs_, image_file_, raw_) in enumerate(zip(np_kp_pairs_pos, image_file_pos, raw_pos)): 
    time_start = time.time()
    pairs_temp = pairs_[None, :, :]

    p12 = pairwise_angles(pairs_temp, np_kp_pairs_neg)
    p12_sum = np.sum(p12, axis=-1)
    p12_argmin = np.argmin(p12_sum, axis=-1)

    # kp_angle_raw_pos_temp_ = kp_coco_angles_pos[idx_pos]
    kp_angle_raw_neg_temp_ = kp_coco_angles_neg[p12_argmin]
    
    img_ = mmcv.imread(image_file_neg[p12_argmin])
    image_pos_temp = mmcv.imread(image_file_)

    print (p12_argmin)
    temp_raw_neg = raw_neg[p12_argmin]['annotations'][0]['keypoints']
    temp_raw_pos = raw_['annotations'][0]['keypoints']

    temp_raw_neg, temp_raw_pos = np.array(temp_raw_neg), np.array(temp_raw_pos)  

    # pos_face, pos_torso, pos_leg = split_in_face_legs_torso(temp_raw_pos)
    # neg_face, neg_torso, neg_leg = split_in_face_legs_torso(temp_raw_neg)
    # temp_pos_transform_face = transform(pos_face, neg_face)
    # temp_pos_transform_torso = transform(pos_torso, neg_torso)
    # temp_pos_transform_leg = transform(pos_leg, neg_leg)
    # temp_pos_transform = np.vstack((temp_pos_transform_face, \
    #                                 temp_pos_transform_torso, \
    #                                 temp_pos_transform_leg))

    pos_face, pos_torso = split_in_face_torso(temp_raw_pos)
    neg_face, neg_torso = split_in_face_torso(temp_raw_neg)
    temp_pos_transform_face = transform(pos_face, neg_face)
    temp_pos_transform_torso = transform(pos_torso, neg_torso)

    temp_pos_transform = np.vstack((temp_pos_transform_face, \
                                    temp_pos_transform_torso))

    # temp_pos_transform = transform(temp_raw_pos, temp_raw_neg)

    angles_transform, angles_draw_transform, angles_coor_transform, \
        angles_name_transform, kp_coco_angles_transform = pair_angles(temp_pos_transform, dict_angles)
    
    pair_angles_ = angles_transform - np_angle_neg[p12_argmin]
    display_angles_transform = np.abs(pair_angles_) >= 20
    print (display_angles_transform)
    
    display_kp_id_transform = []
    for idx, (a, b, c) in enumerate(dict_angles):
        if display_angles_transform[idx]:
            display_kp_id_transform.extend([id_classnames[a], id_classnames[b], id_classnames[c]])
    print ("display_kp_id_transform", display_kp_id_transform)

    display_angles = [True, True, True, True, True, True, True, True, True, True, True, True]
    display_kp_id = []
    for idx, (a, b, c) in enumerate(dict_angles):
        if display_angles[idx]:
            display_kp_id.extend([id_classnames[a], id_classnames[b], id_classnames[c]])
    print ("display_kp_id", display_kp_id)

    img_copy = img_.copy()
    img_copy = draw_transform(img_copy, \
                              [display_angles_transform, display_angles_transform], \
                            [True, False], \
                              [temp_raw_neg, temp_pos_transform], \
                              [kp_angle_raw_neg_temp_, kp_coco_angles_transform], \
                              [display_kp_id, display_kp_id_transform], \
                              [display_angles, display_angles_transform], \
                              skeleton, \
                              [pose_kpt_color, pose_kpt_color_correct], \
                              [pose_limb_color, pose_limb_color_correct])

    image_pos_temp = show_result_angles_orin(image_pos_temp,
                        raw_pos[idx_pos]['annotations'],
                        skeleton,
                        classnames,
                        angles_list=kp_coco_angles_pos[idx_pos],
                        pose_kpt_color=pose_kpt_color_correct,
                        pose_limb_color=pose_limb_color_correct)

    # image_pos_temp = image_pos_temp[100:, 440:650, :]

    image_pos_temp = image_pos_temp[150:-20, 440:650, :]
    img_copy = img_copy[:457, 340:555, :]

    img_draw_neg = cv2.resize(img_copy, (int(w), int(h)))
    img_draw_pos = cv2.resize(image_pos_temp, (int(w), int(h)))
    img_draw_pos = put_instruction(img_draw_pos, display_angles_transform, \
                                   angles_transform, np_angle_neg[p12_argmin], pair_angles_)
    img_draw = np.hstack((img_draw_pos, img_draw_neg))

    mmcv.imwrite(img_draw, "{}.jpg".format(idx_pos))

def show_result_angles_orin(img,
                        result,
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

        img = mmcv.imread(img)
        img = img.copy()
        img_h, img_w, _ = img.shape

        bbox_result = []
        pose_result = []
        for res in result:
            bbox_result.append(res['bbox'])
            pose_result.append(res['keypoints'])

        if len(bbox_result) > 0:
            bboxes = np.vstack(bbox_result)
            for kpts in pose_result:
                # draw each point on image
                if pose_kpt_color is not None:
                    kpts = kpts[:17]
                    print ("len(kpts)", len(kpts))
                    print ("len(pose_kpt_color)", len(pose_kpt_color))
                    assert len(pose_kpt_color) == len(kpts)
                    for kid, kpt in enumerate(kpts):
                        if kid in [0, 1, 2, 3, 4]:
                            continue
                        x_coord, y_coord, kpt_score = int(kpt[0]), int(
                            kpt[1]), kpt[2]
                        if kpt_score > kpt_score_thr:
                            img_copy = img.copy()
                            r, g, b = pose_kpt_color[kid]
                            cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                                       radius, (int(r), int(g), int(b)), -1)
                            # if classnames:
                            #     cv2.putText(img_copy, "{}".format(classnames[kid]),(int(x_coord), int(y_coord)),
                            #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (int(r), int(g), int(b)), 1)
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
                        if any(i in [1, 2, 3, 4, 5]for i in sk):
                            continue
                        pos1 = (int(kpts[sk[0] - 1][0]), int(kpts[sk[0] - 1][1]))
                        pos2 = (int(kpts[sk[1] - 1][0]), int(kpts[sk[1] - 1][1]))
                        middle12 = (int((pos1[0] + pos2[0])/2), int((pos1[1] + pos2[1])/2))
                        if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                                and pos1[1] < img_h and pos2[0] > 0
                                and pos2[0] < img_w and pos2[1] > 0
                                and pos2[1] < img_h
                                and kpts[sk[0] - 1][2] > kpt_score_thr
                                and kpts[sk[1] - 1][2] > kpt_score_thr):
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
                                min(1, 0.5 *
                                    (kpts[sk[0] - 1][2] + kpts[sk[1] - 1][2])))
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)
            # for angle, sub_angle, list_coor, (na, nb, nc) in angles_list:
            #     r, g, b = pose_kpt_color[id_classnames[nb]]
            #     all_x = 0
            #     all_y = 0
            #     list_coor = np.array(list_coor)
            #     plot_coor = np.average(list_coor, axis=0, weights=[0.1, 0.8, 0.1])

            #     from PIL import Image, ImageDraw
            #     #convert image opened opencv to PIL
            #     img_pil = Image.fromarray(img)
            #     draw = ImageDraw.Draw(img_pil)
            #     #draw angle
            #     sub1_LKnee_Angle, sub2_LKnee_Angle = sub_angle
            #     ca, cb, cc = list_coor
            #     shape_LKnee = [(cb[0] - 15, cb[1] - 15), (cb[0] + 15, cb[1] + 15)]
            #     draw.arc(shape_LKnee, start=sub2_LKnee_Angle, end=sub1_LKnee_Angle, fill=(r, g, b))
            #     img = np.array(img_pil)
            #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            #     cv2.putText(img, "{}".format(str(round(angle, 2))), (int(plot_coor[0]), int(plot_coor[1])),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

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
                        result,
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

        img = mmcv.imread(img)
        img = img.copy()
        img_h, img_w, _ = img.shape

        bbox_result = []
        pose_result = []
        for res in result:
            bbox_result.append(res['bbox'])
            pose_result.append(res['keypoints'])

        if len(bbox_result) > 0:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            # mmcv.imshow_bboxes(
            #     img,
            #     bboxes,
            #     colors=bbox_color,
            #     top_k=-1,
            #     thickness=thickness,
            #     show=False,
            #     win_name=win_name,
            #     wait_time=wait_time,
            #     out_file=None)

            for kpts in pose_result:
                # draw each point on image
                if pose_kpt_color is not None:
                    kpts = kpts[:17]
                    # print ("len(kpts)", len(kpts))
                    # print ("len(pose_kpt_color)", len(pose_kpt_color))
                    assert len(pose_kpt_color) == len(kpts)
                    for kid, kpt in enumerate(kpts):
                        if kid in [0, 1, 2, 3, 4]:
                            continue
                        x_coord, y_coord, kpt_score = int(kpt[0]), int(
                            kpt[1]), kpt[2]
                        if kpt_score > kpt_score_thr:
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
                        if any(i in [1, 2, 3, 4, 5]for i in sk):
                            continue
                        pos1 = (int(kpts[sk[0] - 1][0]), int(kpts[sk[0] - 1][1]))
                        pos2 = (int(kpts[sk[1] - 1][0]), int(kpts[sk[1] - 1][1]))
                        middle12 = (int((pos1[0] + pos2[0])/2), int((pos1[1] + pos2[1])/2))
                        if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                                and pos1[1] < img_h and pos2[0] > 0
                                and pos2[0] < img_w and pos2[1] > 0
                                and pos2[1] < img_h
                                and kpts[sk[0] - 1][2] > kpt_score_thr
                                and kpts[sk[1] - 1][2] > kpt_score_thr):
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
                                min(1, 0.5 *
                                    (kpts[sk[0] - 1][2] + kpts[sk[1] - 1][2])))
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
                # vis_img = show_result_angles(img,
                #     anno['annotations'],
                #     skeleton,
                #     classnames,
                #     angles_list=kp_coco_angles,
                #     pose_kpt_color=pose_kpt_color,
                #     pose_limb_color=pose_limb_color)
                # cv2.imwrite('{}.jpg'.format(file_name_jpg), vis_img)
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

np_kp_pos, raw_pos, image_file_pos, np_angle_pos, \
    np_angle_draw_pos, kp_angle_coor_pos, kp_angle_name_pos, \
        kp_angle_raw_pos, kp_coco_angles_pos, np_kp_pairs_pos = vectorizer_kp(folder_pos)
np_kp_neg, raw_neg, image_file_neg, np_angle_neg, \
    np_angle_draw_neg, kp_angle_coor_neg, kp_angle_name_neg, \
        kp_angle_raw_neg, kp_coco_angles_neg, np_kp_pairs_neg = vectorizer_kp(folder_neg)

compare_np_angle_draw_neg = np_angle_draw_neg[:1, :, :]
compare_kp_pairs_pos = np_kp_pairs_pos[:10, :, :]


def compare_kp(comp, pos_frame_kps):
    idx_match = 0
    for idx, pos_frame_kp in enumerate(pos_frame_kps):
        count_match = 0  
        for part_angle, part_angle_comp in zip(pos_frame_kp, comp):
            alpha, beta = part_angle
            alpha_comp, beta_comp = part_angle_comp
            min_angle1 = min(alpha_comp, alpha)
            max_angle1 = max(alpha_comp, alpha)
            min_angle2 = min(beta_comp, beta)
            max_angle2 = max(beta_comp, beta)
            if min_angle1 + 20 > max_angle1 and min_angle2 + 20 > max_angle2:
                count_match += 1
        if count_match >= 11:
            idx_match = idx
    return idx_match

def pairwise_angles(v1, v2):
    acos_invalue = np.sum(v1*v2, axis=-1) / (LA.norm(v1, axis=-1) * LA.norm(v2, axis=-1))
    acos_value = np.arccos(np.round(acos_invalue, 2))
    degrees_value = np.degrees(acos_value)
    where_are_nans = np.isnan(degrees_value)
    degrees_value[where_are_nans] = 20000
    return degrees_value

def pairwise_part(v1, v2, threshold):
    acos_invalue = np.sum(v1*v2, axis=-1) / (LA.norm(v1, axis=-1) * LA.norm(v2, axis=-1))
    acos_value = np.arccos(np.round(acos_invalue, 2))
    degrees_value = np.degrees(acos_value)
    where_are_nans = np.isnan(degrees_value)
    degrees_value[where_are_nans] = 20000
    return idx_match

threshold = 10
fps = 30
w, h = 477, 900
size = (w*5, h)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter("./out_{}.mp4".format(time.time()), fourcc, fps, size)
for idx_pos, (pairs_, image_file_, raw_) in enumerate(zip(np_kp_pairs_pos, image_file_pos, raw_pos)): 
    time_start = time.time()
    pairs_temp = pairs_[None, :, :]

    p12 = pairwise_angles(pairs_temp, np_kp_pairs_neg)
    p12_sum = np.sum(p12, axis=-1)
    p12_argmin = np.argmin(p12_sum, axis=-1)
    print (p12_argmin)

    p12_abs = (np_angle_pos[idx_pos] - np_angle_neg[p12_argmin])
    p12_threshold = np.logical_or(p12_abs > threshold, p12_abs < -threshold)
    # p12_count = np.count_nonzero(p12_threshold, axis=-1)
    # p12_argmin = np.argmin(p12_count, axis=-1)

    kp_angle_raw_pos_temp_ = kp_coco_angles_pos[idx_pos]
    kp_angle_raw_neg_temp_ = kp_coco_angles_neg[p12_argmin]

    diff_angle_pos = p12_abs
    diff_angle_pos_bool = p12_threshold

    image_pos_temp = mmcv.imread(image_file_)
    image_neg_temp = mmcv.imread(image_file_neg[p12_argmin])

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
                        raw_neg[p12_argmin]['annotations'],
                        skeleton,
                        classnames,
                        diff_angle_neg_threshold=diff_angle_pos_bool,
                        diff_angle=diff_angle_pos,
                        angles_list=kp_angle_raw_neg_temp_,
                        pose_kpt_color_bool=color_kpt_bool,
                        pose_kpt_color=palette[color_kpt],
                        pose_limb_color=palette[color_limb])

    img_draw_temp = show_result_angles_orin(img_draw_temp,
                    raw_neg[p12_argmin]['annotations'],
                    skeleton,
                    classnames,
                    angles_list=kp_angle_raw_neg_temp_,
                    pose_kpt_color=pose_kpt_color,
                    pose_limb_color=pose_limb_color)

    # image_pos_temp = image_pos_temp
    image_pos_temp = show_result_angles_orin(image_pos_temp,
                    raw_pos[idx_pos]['annotations'],
                    skeleton,
                    classnames,
                    angles_list=kp_angle_raw_pos_temp_,
                    pose_kpt_color=pose_kpt_color,
                    pose_limb_color=pose_limb_color)

    img_draw_neg = cv2.resize(img_draw_temp, (w, h))
    img_draw_pos = cv2.resize(image_pos_temp, (w*4, h))

    img_draw = np.hstack((img_draw_neg, img_draw_pos))

    if img_draw is not None:
        mmcv.imwrite(img_draw, "./{}".format(os.path.basename(image_file_)))

    # print ("time_end: ", time.time() - time_start, p12_argmin)
    videoWriter.write(img_draw)

videoWriter.release()

import os

threshold = 30
fps = 30
size = (477*5, 900)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter("./out_{}.mp4".format(time.time()), fourcc, fps, size)
for idx_pos, (pairs_, image_file_, raw_) in enumerate(zip(np_angle_draw_pos, image_file_pos, raw_pos)): 
    time_start = time.time()
    pairs_temp = pairs_[None, :, :]
    
    p12_abs = np.absolute((pairs_temp - np_angle_draw_neg).sum(-1))
    p12_threshold = p12_abs > threshold
    p12_count = np.count_nonzero(p12_threshold, axis=-1)
    p12_argmin = np.argmin(p12_count, axis=-1)

    kp_angle_raw_pos_temp_ = kp_coco_angles_pos[idx_pos]
    kp_angle_raw_neg_temp_ = kp_coco_angles_neg[p12_argmin]

    diff_angle_pos = p12_abs[p12_argmin]
    diff_angle_pos_bool = p12_threshold[p12_argmin]

    image_pos_temp = mmcv.imread(image_file_)
    image_neg_temp = mmcv.imread(image_file_neg[p12_argmin])

    color_kpt = np.array([
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16
    ])
    color_limb = np.array([
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16
    ])

    color_kpt_bool = np.array([
        False, False, False, False, False, False, False, False,  
        False, False, False, False, False, False, False, False, False
    ])

    for idx, (angle_diff , angle_bool) in enumerate(zip(diff_angle_pos, diff_angle_pos_bool)):
         if angle_bool:
            list_angle_name = dict_angles[idx]
            for angle_name in list_angle_name:
                id_class_kpt = id_classnames[angle_name]
                color_kpt_bool[id_class_kpt] = True

    img_draw_temp = show_result_angles(image_neg_temp,
                        raw_neg[p12_argmin]['annotations'],
                        skeleton,
                        classnames,
                        diff_angle_neg_threshold=diff_angle_pos_bool,
                        diff_angle=diff_angle_pos,
                        angles_list=kp_angle_raw_neg_temp_,
                        pose_kpt_color_bool=color_kpt_bool,
                        pose_kpt_color=palette[color_kpt],
                        pose_limb_color=palette[color_limb])

    image_pos_temp = image_pos_temp
    # image_pos_temp = show_result_angles_orin(image_pos_temp,
    #                                         raw_pos[idx_pos]['annotations'],
    #                                         skeleton,
    #                                         classnames,
    #                                         angles_list=kp_angle_raw_pos_temp_,
    #                                         pose_kpt_color=pose_kpt_color,
    #                                         pose_limb_color=pose_limb_color)

    img_draw_neg = cv2.resize(img_draw_temp, (477, 900))
    img_draw_pos = cv2.resize(image_pos_temp, (477*4, 900))

    img_draw = np.hstack((img_draw_neg, img_draw_pos))

    if img_draw is not None:
        mmcv.imwrite(img_draw, "./{}.jpg".format(idx_pos))

    # print ("time_end: ", time.time() - time_start, p12_argmin)
    videoWriter.write(img_draw)

videoWriter.release()


threshold = 50
fps = 30
size = (477*5, 900)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter("./out_{}.mp4".format(time.time()), fourcc, fps, size)
min_pose_idx = [(0, 0, None)]*8
for idx_neg, (pairs_, image_file_, raw_) in enumerate(zip(np_angle_draw_neg, image_file_neg, raw_neg)): 
    time_start = time.time()
    pairs_temp = pairs_[None, :, :]
    
    p12 = np.absolute((pairs_temp - np_angle_draw_pos).sum(-1)) < threshold
    p12 = np.count_nonzero(p12, axis=-1)
    p12_argmin = np.argmax(p12, axis=-1)

    # p12 = pairwise_angles(pairs_temp, np_kp_pairs_pos)
    # p12_sum = np.sum(p12, axis=-1)
    # p12_argmin = np.argmin(p12_sum, axis=-1)

    kp_angle_raw_neg_temp_ = kp_coco_angles_neg[idx_neg]
    kp_angle_raw_pos_temp_ = kp_coco_angles_pos[p12_argmin]

    np_angle_neg_temp_ = np_angle_neg[idx_neg]
    np_angle_pos_temp_ = np_angle_pos[p12_argmin]
    diff_angle_neg = np_angle_neg_temp_ - np_angle_pos_temp_

    image_neg_temp = mmcv.imread(image_file_)
    image_pos_temp = mmcv.imread(image_file_pos[p12_argmin])

    color_kpt = np.array([
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16
    ])
    color_limb = np.array([
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16
    ])

    for idx, (angle_diff , angle_bool) in enumerate(zip(diff_angle_pos, diff_angle_pos_bool)):
         if angle_bool:
            list_angle_name = dict_angles[idx]
            print (list_angle_name)
            first, middle, last = list_angle_name
            id_class_kpt = id_classnames[middle]
            color_kpt_bool[id_class_kpt] = True

    # diff_angle_neg_threshold = (np.absolute(diff_angle_neg) > threshold)
    # for angle_neg, angle_dict in zip(diff_angle_neg_threshold, dict_angles):
    #      if angle_neg:
    #         angle_neg_list = dict_angles
    #         for angle_name in angle_neg_list:
    #             first, middle, last = angle_name
    #             color_kpt[id_classnames[middle]] = 17
    #             for idx_pair, sk_pair in enumerate(skeleton):
    #                 if id_classnames[middle] in sk_pair:
    #                     color_limb[idx_pair] = 17
    #             # for kpt_name in angle_name:
    #             #     print (kpt_name)
    #             #     # color_limb[id_classnames[kpt_name]] = 17
    # # print (color_kpt)

    img_draw_temp = show_result_angles(image_neg_temp,
                        raw_neg[idx_neg]['annotations'],
                        skeleton,
                        classnames,
                        diff_angle_neg_threshold=diff_angle_neg_threshold,
                        diff_angle=diff_angle_neg,
                        angles_list=kp_angle_raw_neg_temp_,
                        pose_kpt_color=palette[color_kpt],
                        pose_limb_color=palette[color_limb])

    img_draw_neg = cv2.resize(img_draw_temp, (477, 900))
    img_draw_pos = cv2.resize(image_pos_temp, (477*4, 900))

    img_draw = np.hstack((img_draw_neg, img_draw_pos))

    # cur_min_idx, cur_min_value, _ = min_pose_idx[p12_argmin]
    # if cur_min_value <= p12_sum[p12_argmin]:
    #     min_pose_idx[p12_argmin] = (p12_argmin, p12_sum[p12_argmin], img_draw)

    # print ("time_end: ", time.time() - time_start, p12_argmin)
    videoWriter.write(img_draw)

videoWriter.release()

for p12_idx, p12_sum, img_draw in min_pose_idx:
    if img_draw is not None:
        mmcv.imwrite(img_draw, "./{}.jpg".format(p12_idx))


threshold = 20
fps = 30
size = (477*5, 900)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter("./out_{}.mp4".format(time.time()), fourcc, fps, size)

for idx_neg, (pairs_, image_file_, raw_) in enumerate(zip(np_kp_pairs_neg, image_file_neg, raw_neg)): 
    time_start = time.time()
    pairs_temp = pairs_[None, :, :]
    p12 = pairwise_angles(pairs_temp, np_kp_pairs_pos)
    p12_sum = np.sum(p12, axis=-1)
    p12_argmin = np.argmin(p12_sum, axis=-1)


    np_angle_neg_temp_ = np_angle_neg[idx_neg]
    np_angle_pos_temp_ = np_angle_pos[p12_argmin]
    diff_angle_neg = np_angle_neg_temp_ - np_angle_pos_temp_

    p12_abs = (np_angle_draw_pos[idx_pos] - np_angle_draw_neg).sum(-1)
    p12_threshold = p12_abs > threshold
    # p12_count = np.count_nonzero(p12_threshold, axis=-1)
    # p12_argmin = np.argmin(p12_count, axis=-1)

    diff_angle_pos = p12_abs[p12_argmin]
    diff_angle_pos_bool = p12_threshold[p12_argmin]

    image_neg_temp = mmcv.imread(image_file_)
    image_pos_temp = mmcv.imread(image_file_pos[p12_argmin])

    color_kpt = np.array([
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16
    ])
    color_limb = np.array([
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16
    ])
    color_kpt_bool = np.array([
        False, False, False, False, False, False, False, False,  
        False, False, False, False, False, False, False, False, False
    ])

    for idx, (angle_diff , angle_bool) in enumerate(zip(diff_angle_pos, diff_angle_pos_bool)):
         if angle_bool:
            list_angle_name = dict_angles[idx]
            first, middle, last = list_angle_name
            id_class_kpt = id_classnames[middle]
            color_kpt_bool[id_class_kpt] = True


    # img_draw_temp = image_neg_temp
    img_draw_temp = show_result_angles(image_neg_temp,
                        raw_neg[idx_neg]['annotations'],
                        skeleton,
                        classnames,
                        diff_angle_neg_threshold=diff_angle_pos_bool,
                        diff_angle=diff_angle_neg,
                        angles_list=kp_angle_raw_neg_temp_,
                        pose_kpt_color_bool=color_kpt_bool,
                        pose_kpt_color=palette[color_kpt],
                        pose_limb_color=palette[color_limb])

    img_draw_neg = cv2.resize(img_draw_temp, (477, 900))
    img_draw_pos = cv2.resize(image_pos_temp, (477*4, 900))

    img_draw = np.hstack((img_draw_neg, img_draw_pos))

    # print ("time_end: ", time.time() - time_start, p12_argmin)
    videoWriter.write(img_draw)

videoWriter.release()


def show_result(img,
                result,
                skeleton=None,
                classnames=None,
                kpt_score_thr=0.0,
                bbox_color='green',
                pose_kpt_color=None,
                pose_limb_plot=None,
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

        img = mmcv.imread(img)
        img = img.copy()
        img_h, img_w, _ = img.shape
        not_include = [0, 1, 2, 3, 4]

        bbox_result = []
        pose_result = []
        for res in result:
            bbox_result.append(res['bbox'])
            pose_result.append(res['keypoints'])

        if len(bbox_result) > 0:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            # mmcv.imshow_bboxes(
            #     img,
            #     bboxes,
            #     colors=bbox_color,
            #     top_k=-1,
            #     thickness=thickness,
            #     show=False,
            #     win_name=win_name,
            #     wait_time=wait_time,
            #     out_file=None)

            for _, (kpts, angles) in enumerate(zip(pose_result, angles_list)):
                # draw each point on image
                if pose_kpt_color is not None:
                    kpts = kpts[:17]
                    assert len(pose_kpt_color) == len(kpts)
                    for kid, kpt in enumerate(kpts):
                        if kid in not_include:
                            continue
                        x_coord, y_coord, kpt_score = int(kpt[0]), int(
                            kpt[1]), kpt[2]
                        if kpt_score > kpt_score_thr:
                            img_copy = img.copy()
                            r, g, b = pose_kpt_color[kid]
                            cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                                       radius, (int(r), int(g), int(b)), -1)
                            if classnames:
                                cv2.putText(img_copy, "{}".format(classnames[kid]),(int(x_coord), int(y_coord)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (int(r), int(g), int(b)), 1)
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
                        if pose_limb_plot[sk_id]:
                            if any(i in not_include for i in sk):
                                continue
                            pos1 = (int(kpts[sk[0] - 1][0]), int(kpts[sk[0] - 1][1]))
                            pos2 = (int(kpts[sk[1] - 1][0]), int(kpts[sk[1] - 1][1]))
                            middle12 = (int((pos1[0] + pos2[0])/2), int((pos1[1] + pos2[1])/2))
                            if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                                    and pos1[1] < img_h and pos2[0] > 0
                                    and pos2[0] < img_w and pos2[1] > 0
                                    and pos2[1] < img_h
                                    and kpts[sk[0] - 1][2] > kpt_score_thr
                                    and kpts[sk[1] - 1][2] > kpt_score_thr):
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
                                if len(angles) != 0:
                                    cv2.putText(img_copy, "{}".format(str(round(angles[sk_id], 2))), (middle12[0], middle12[1]),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (int(r), int(g), int(b)), 1)
                                cv2.fillConvexPoly(img_copy, polygon,
                                                (int(r), int(g), int(b)))
                                transparency = max(
                                    0,
                                    min(1, 0.5 *
                                        (kpts[sk[0] - 1][2] + kpts[sk[1] - 1][2])))
                                cv2.addWeighted(
                                    img_copy,
                                    transparency,
                                    img,
                                    1 - transparency,
                                    0,
                                    dst=img)
                        
                    idx_left_x = 30
                    idx_left_h = img_h - int(0.1*img_h)
                    for sk_id, sk in enumerate(skeleton):
                        if pose_limb_plot[sk_id]:
                            r, g, b = pose_limb_color[sk_id]
                            kp1, kp2 = skeleton[sk_id]
                            display_text = "{} - {}: {}".format(classnames[kp1-1], \
                                classnames[kp2-1], str(round(angles[sk_id], 2)))
                            # cv2.putText(img, "{}".format(str(round(angles[sk_id], 2))), (idx_left_x, idx_left_h),
                            #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (int(r), int(g), int(b)), 2)
                            cv2.putText(img, "{}".format(display_text), (idx_left_x, idx_left_h),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (int(r), int(g), int(b)), 2)
                            idx_left_h = idx_left_h + 15

        return img