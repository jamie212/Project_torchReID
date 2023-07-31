from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from PIL import Image
from scipy.optimize import linear_sum_assignment
import cv2
from natsort import natsorted
# import yh as yh
import new_yh as yh
from tqdm import tqdm, trange
import sys
sys.path.append("deep_person_reid")
from torchreid.utils import FeatureExtractor
import json
from config import parse_arguments

extractor = FeatureExtractor(
    # model_name='osnet_x1_0',
    # model_path='log/osnet/osnet_x1_0_imagenet.pth',
    # device='cuda'
    # model_name='resnet50',
    # model_path='log/resnet/model.pth.tar-60',
    # device='cuda'
    model_name='resnext101_32x8d',
    model_path='log/resnext101/model/model.pth.tar-60',
    device='cuda'
)

def cos_sim(a, b):
    a = np.mat(a)
    b = np.mat(b)
    return float(a * b.T) / (np.linalg.norm(a) * np.linalg.norm(b))

# def load_config():
#     with open('config.json') as config_file:
#         data = json.load(config_file)
#     return data

# 以下是演算法組寫的（以上是reID model相關）
def get_iou(box1, box2):
    return yh.iou(box1, box2)

def check_sizediff(box1, box2): 
    if box1[2]/box2[2] > 2 or box2[2]/box1[2] > 2 or box1[3]/box2[3] > 2 or box2[3]/box1[3] > 2:
        return True

def get_similarity(box1, box1_rgb, box2, box2_rgb):
    height, width, _ = box1_rgb.shape
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    ul, lr = yh.dilate_bbox(box1, height, width)
    # ul = (box1[0], box1[1])
    # lr = (box1[0] + box1[2], box1[1] + box1[3])
    y_cropped_img = box1_rgb[ul[1]:lr[1], ul[0]:lr[0]]
    # print(ul, lr)

    ul, lr = yh.dilate_bbox(box2, height, width)
    # ul = (box2[0], box2[1])
    # lr = (box2[0] + box2[2], box2[1] + box2[3])
    x_cropped_img = box2_rgb[ul[1]:lr[1], ul[0]:lr[0]]
    # print(ul, lr)

    img1 = y_cropped_img
    img2 = x_cropped_img

    image_list = [
        img1,img2
    ]

    features = extractor(image_list)
    features = features.cpu()
    # print(cos_sim(features[0],features[1]))
    sim_score = cos_sim(features[0],features[1])
    # tensor_y_cropped_img, tensor_x_cropped_img = transform(y_cropped_img).unsqueeze(0), transform(x_cropped_img).unsqueeze(0)
    # sim_score = similarity(tensor_y_cropped_img, tensor_x_cropped_img, model)

    return sim_score

def get_moved_dist(x_box, y_box):
    y_midpoint_x = y_box[0] + y_box[2]/2
    y_midpoint_y = y_box[1] + y_box[3]/2
    x_midpoint_x = x_box[0] + x_box[2]/2
    x_midpoint_y = x_box[1] + x_box[3]/2
    dist = math.sqrt((x_midpoint_x - y_midpoint_x)**2 + (x_midpoint_y - y_midpoint_y)**2)
    return dist


def hungarian_algorithm(cost_matrix):
    n, m = cost_matrix.shape
    matches = np.full(n, -1, dtype=int)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    for i in range(len(row_ind)):
        matches[row_ind[i]] = col_ind[i]
    return matches

def check_covered(z_box, x_box):
    overlap_left_x = max(z_box[0], x_box[0])
    overlap_left_y = max(z_box[1], x_box[1])
    overlap_right_x = min(z_box[0] + z_box[2], x_box[0] + x_box[2])
    overlap_right_y = min(z_box[1] + z_box[3], x_box[1] + x_box[3])

    overlap_area = (overlap_right_x - overlap_left_x) * (overlap_right_y - overlap_left_y) if overlap_left_x < overlap_right_x and overlap_left_y < overlap_right_y else 0

    min_area = min((z_box[2] * z_box[3]), (x_box[2] * x_box[3]))

    return overlap_area / min_area

def check_in_yolo(yolo_box, x_box):  # return True means detected by yolo --> not garbage / False --> maybe garbage
    # print(yolo_box) 
    for box in yolo_box:
        z_box = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
        print('check covered : {}'.format(check_covered(z_box, x_box)))
        if check_covered(z_box, x_box) > 0.7:
            print(f'yolo z: {z_box}')
            print(f'inte x: {x_box}')
            return True
    return False

def add_trackinglist(wait_to_add, y_box, current_frame_idx):
    tmp = {
        "bbox": y_box,
        "frame_idx": current_frame_idx,
        "ref": True,
        "duration": 1,
        "state": 0,
        "no_pair_counter": 0, 
        "is_garbage": False,
        "moved_dist" : 0,
        "check_yolo_cnt" : 0,  # number of times that has checked in yolo, if == 3 --> check not_in_yolo
        "not_in_yolo" : 0,
        "be_covered" : 0, 
        "appear_bbox" : y_box
        # "exist": True,
        # "not_moving_cnt" : 0, 
        # "vanish_counter": 0
    }
    wait_to_add.append(tmp)
    return wait_to_add

def update_trackinglist(trackinglist_info, y_box):
    x_box = trackinglist_info["bbox"]
    trackinglist_info["moved_dist"] += get_moved_dist(x_box, y_box)
    trackinglist_info["bbox"] = y_box
    trackinglist_info["ref"] = True
    trackinglist_info["duration"] += 1
    trackinglist_info["state"] = 1
    trackinglist_info["no_pair_counter"] = 0
    trackinglist_info["be_covered"] = 0
    # trackinglist_info["exist"] = True
    # trackinglist_info["vanish_counter"] = 0

    return trackinglist_info


def main(args):
    '''
        ./our_datum
            ├ input                         <----- path2input
            │   ├ video_1.mp4               <----- rgb_cap
            │   ├ ...
            │   └ video_n.mp4
            ├ tmp
            │   ├ videos                    <----- path2processed_videos
            │   │   ├ video_1.mp4
            │   │   ├ video_2.mp4
            │   │   ├ ...
            │   │   └ video_n.mp4
            │   └ npys                      <----- path2yolo_bbox_folders
            │       ├ video_1               <----- path2video_yolo_bbox
            │       │   ├ frame_0.npy
            │       │   ├ frame_0.npy
            │       │   ├ ...   
            │       │   └ frame_n.npy
            │       ├ ...
            │       └ video_n
            │           ├ frame_0.npy
            │           ├ frame_0.npy
            │           ├ ...   
            │           └ frame_n.npy
            └ output
                ├ video_1                   <----- path2output
                │   ├ rgb
                │   │   ├ frame_0.png
                │   │   ├ ...
                │   │   └ frame_m.png
                │   │   
                │   └ mask
                │       ├ frame_0.png
                │       ├ ...
                │       └ frame_m.png
                ├ ...
                └ video_n
    '''
    # path2input, path2mask_videos, path2yolo_bbox_folders = "./our_datum_0516/input", "./our_datum_0516/tmp/videos", "./our_datum_0516/tmp/npys"
    path2input, path2mask_videos, path2yolo_bbox_folders = "./datum_0724/input", "./datum_0724/tmp/videos", "./datum_0724/tmp/npys"

    if not os.path.exists(path2input): 
        print(f"Fail to get any input. Kill the process.")
        os._exit(0)

    if not os.path.exists(path2mask_videos):
        print(f"Fail to get mask_videos. Kill the process.")
        os._exit(0)

    if not os.path.exists(path2yolo_bbox_folders):
        print(f"Fail to get yolo_bbox_folders. Kill the process.")
        os._exit(0)

    rgb_videos, mask_videos, yolo_bbox_folders = natsorted(os.listdir(path2input)), natsorted(os.listdir(path2mask_videos)), natsorted(os.listdir(path2yolo_bbox_folders))

    # config = load_config() 
    
    # 建立模型
    # model = load_model()
    # transform = transforms.Compose([
    #     transforms.ToTensor()
    # ])s

    # Draw bbox
    # 線的厚度, 類型
    thickness, lineType, font = 2, 4, cv2.FONT_HERSHEY_SIMPLEX

    # 影片每幀Resize長寬
    resize_width, resize_height = 960, 540
    # resize_width, resize_height = 1920, 1080

    check_yolo_for_ghost = [30, 300, 1800, 3600, 5400]

    '''
        設定影片播放範圍
        videos_range[0] = (500, 700) 代表處理第一部影片的500到700幀s
        videos_range[1] = (0, 0)     代表處理第二部影片的所有幀
        videos_range[2] = (1, 10)    代表處理第二部影片的所有幀
        如果有第四部影片，卻只設定三部影片的範圍，會對第四部影片的每一幀都做處理
    '''
    videos_need_process = [4]

    videos_range = [] # 此處設定影片播放範圍處，從第0幀開始算

    need_append = len(rgb_videos) - len(videos_range)
    if need_append<0:
        print(f"The playback time of the video is set incorrectly. Kill the process.")
        os._exit(0)
    elif need_append>0:
        for idx in range(need_append): videos_range.append((0, 0))

    for video_idx, video in enumerate(rgb_videos):
        # print('test')

        # 設定哪幾部影片要處理
        if not video_idx in videos_need_process : continue       

        # 設定輸出資料夾
        path2output = f"./datum_0724/output/{video_idx}"

        # path2output = f"./datum_0711/output/{video_idx}"
        if not os.path.exists(f"{path2output}/mask"): os.makedirs(f"{path2output}/mask")
        if not os.path.exists(f"{path2output}/rgb"): os.makedirs(f"{path2output}/rgb")

        # 讀取影片
        rgb_video, mask_video = f"{path2input}/{rgb_videos[video_idx]}", f"{path2mask_videos}/{mask_videos[video_idx]}"
        rgb_cap = cv2.VideoCapture(rgb_video) # 彩色影片，讀取彩幀用
        mask_cap = cv2.VideoCapture(mask_video)

        # 檢查是否成功打開影片
        if not rgb_cap.isOpened(): 
            print(f"Fail to open: {rgb_video}. Kill the process.")
            os._exit(0)
        if not mask_cap.isOpened(): 
            print(f"Fail to open: {mask_video}. Kill the process.")
            os._exit(0)

        # 取得幀數, 影片長寬
        total_frames = int( rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT) )
        # width, height = int( rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH) ), int( rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )

        # 設定yolo_bbox路徑        
        path2video_yolo_bbox = f"{path2yolo_bbox_folders}/{yolo_bbox_folders[video_idx]}"
        yolo_bboxs = natsorted(os.listdir(path2video_yolo_bbox))

        # 初始化上一幀
        previous_rgb_frame = None

        # Initialize Tracking List
        tracking_list = []

        # 設定影片範圍
        video_start, video_end = videos_range[video_idx][0], videos_range[video_idx][1]        
        if video_start==0 and video_end==0:
            video_end = total_frames-1
        elif video_start<0 or video_end>total_frames-1 or video_end-video_start<0:
            print(f"video_start of {video} is set incorrectly. Kill the process.")
            os._exit(0)
        
        # video_start, video_end = 10438, 10915
        
        print(f"Processing video: {video_idx}")

        for current_frame_idx in range(total_frames):
            print(f'current_frame_idx: {current_frame_idx}')
            

            rgb_rval, now_rgb_frame = rgb_cap.read()  # 拍攝的彩色圖片
            mask_rval, mask = mask_cap.read() # mask_processed

            # 如果讀取失敗，程式終止
            if not rgb_rval: 
                print(f"Fail to read the {current_frame_idx} frame of {rgb_video}. Kill the process.")
                os._exit(0)
            if not mask_rval: 
                print(f"Fail to read the {current_frame_idx} frame of {mask_video}. Kill the process.")
                os._exit(0)

            # 播放範圍
            if current_frame_idx<video_start: continue
            if current_frame_idx>video_end: break

            # 每一幀Resize, 灰階, 二值化 etc.
            now_rgb_frame, denoise_mask, interest_bboxes = yh.img_processing_update(now_rgb_frame, mask, resize_width, resize_height)
            wait_to_add = [] # wait_to_add 用來儲存加入清單之物體

            ############################## HUNGARIAN ##############################

            # create cost_matrix
            n, m = len(interest_bboxes), len(tracking_list)
            print(f'interest: {n}, tracking: {m}')
            
            if m == 0: # tracking list is empty
                
                for i, y_box in enumerate(interest_bboxes):
                    wait_to_add = add_trackinglist(wait_to_add, y_box, current_frame_idx)
                    
            elif m != 0 and n != 0:
                cost_matrix = np.zeros((n, m))
                for i, y_box in enumerate(interest_bboxes): # y
                    # print(y_box)
                    y_area = y_box[2]*y_box[3]
                    
                    for j, x_box in enumerate(tracking_list):    # x
                        # get iou
                        iou_score = yh.iou(y_box, x_box["bbox"])

                        # get difference of box size
                        x_area = x_box["bbox"][2]*x_box["bbox"][3]
                        size_score = abs(y_area - x_area) / max(y_area, x_area)

                        # get similarity score
                        sim_score = get_similarity(y_box, now_rgb_frame, x_box["bbox"], previous_rgb_frame)

                        # 計算總權重
                        total_score = iou_score + (1 - size_score) + sim_score

                        cost_matrix[i][j] = total_score

                max_cost = np.max(cost_matrix) 
                cost_matrix = max_cost - cost_matrix # 原本是越大越好（因為相似度、iou越大越好），要改成越小越好
                match = hungarian_algorithm(cost_matrix)

                for i, y_box in enumerate(interest_bboxes): 
                    need_update = True
                    print(f'for y_box {i}')
                    if match[i] == -1:        # y比x多，沒有配到x -> 新物體
                        wait_to_add = add_trackinglist(wait_to_add, y_box, current_frame_idx)
                        print("no match, new item")
                    else:                       # 有配到，檢查有沒有過三關
                        print(f'y_box: {y_box}')
                        print(f'match for y: {match[i]}')
                        print(f'x_box: {tracking_list[match[i]]["bbox"]}')
                        print(f'iou: {yh.iou(y_box, tracking_list[match[i]]["bbox"])}')
                        print(f'sizediff: {check_sizediff(y_box, tracking_list[match[i]]["bbox"])}')
                        print(f'similarity: {get_similarity(y_box, now_rgb_frame, tracking_list[match[i]]["bbox"], previous_rgb_frame)}')
                        
                        if get_iou(y_box, tracking_list[match[i]]["bbox"]) > args.high_iou_th and get_similarity(y_box, now_rgb_frame, tracking_list[match[i]]["bbox"], previous_rgb_frame) > args.high_iou_simi_th:
                            print("iou high enough, so similarity can be lower")
                            pass

                        elif (get_iou(y_box, tracking_list[match[i]]["bbox"]) < args.overlap_th) or check_sizediff(y_box, tracking_list[match[i]]["bbox"]) \
                                or (get_similarity(y_box, now_rgb_frame, tracking_list[match[i]]["bbox"], previous_rgb_frame) < args.similarity_th):
                            # 沒過--> 新物體
                            wait_to_add = add_trackinglist(wait_to_add, y_box, current_frame_idx)
                            need_update = False
  
                        else:   # 有配到且通過判斷 --> 更新
                            pass

                        if need_update:
                            if tracking_list[match[i]]["state"] == -2:
                                continue
                            tracking_list[match[i]] = update_trackinglist(tracking_list[match[i]], y_box)
                            # check the updated box is garbage or not
                            print(f'duration: {tracking_list[match[i]]["duration"]}')
                            print(f'move: {tracking_list[match[i]]["moved_dist"]}')

                            if tracking_list[match[i]]["duration"] > args.stay_up_th and tracking_list[match[i]]["moved_dist"] < args.moved_th:
                                tracking_list[match[i]]["check_yolo_cnt"] += 1
                                yolo_bbox = np.load(f"{path2video_yolo_bbox}/{yolo_bboxs[current_frame_idx]}")
                                if check_in_yolo(yolo_bbox, tracking_list[match[i]]["bbox"]) == False:
                                    print("not in yolo")
                                    tracking_list[match[i]]["not_in_yolo"] += 1

                                if tracking_list[match[i]]["check_yolo_cnt"] == 3:
                                    if tracking_list[match[i]]["not_in_yolo"] >= 2:
                                        # maybe garbage or ghost, check the previous yolo box before appear using appear bbox
                                        was_in_yolo = 0
                                        for t in check_yolo_for_ghost:
                                            # check_yolo_idx = current_frame_idx - tracking_list[match[i]]["not_moving_cnt"] + t
                                            check_yolo_idx = tracking_list[match[i]]["frame_idx"] - t
                                            if check_yolo_idx < 0:
                                                check_yolo_idx = 0
                                            print(check_yolo_idx)
                                            pre_yolo_bbox = np.load(f"{path2video_yolo_bbox}/{yolo_bboxs[check_yolo_idx]}")
                                            
                                            if check_in_yolo(pre_yolo_bbox, tracking_list[match[i]]["appear_bbox"]) == True:
                                                print('in checking old yolo:')
                                                print(pre_yolo_bbox)
                                                print(tracking_list[match[i]]["bbox"])
                                                print('old is in yolo')
                                                was_in_yolo += 1
                                            else:
                                                print("old not in yolo")
                                                print(pre_yolo_bbox)
                                                print(tracking_list[match[i]]["bbox"])
                                        if was_in_yolo < 4 :
                                            tracking_list[match[i]]["is_garbage"] = True
                                            tracking_list[match[i]]["state"] = -2
                                            print(tracking_list[match[i]]["state"])
                                            print("is garbage")
                                        else:
                                            print("is ghost")
                                            tracking_list[match[i]]["duration"] = 0
                                        

                                    elif tracking_list[match[i]]["not_in_yolo"] == 0:
                                        tracking_list[match[i]]["duration"] = 0

                                    tracking_list[match[i]]["check_yolo_cnt"] = 0
                                    tracking_list[match[i]]["not_in_yolo"] = 0
          
                            if tracking_list[match[i]]["duration"] > args.stay_up_th / 2:
                                tracking_list[match[i]]["moved_dist"] = 0

                
            remove_idx = []

            # 新增
            for add_obj in wait_to_add:
                tracking_list.append(add_obj)

            for z_idx, z in enumerate(tracking_list): 

                if z["ref"] == True or z["state"] == -2:
                    continue
                tracking_list[z_idx]["state"] = 1
                # 篩掉那些被覆蓋住的
                be_covered = False
                ############
                for x_idx, x in enumerate(tracking_list):
                    if x_idx == z_idx:
                        continue
                    if tracking_list[x_idx]["state"] == -1 or tracking_list[x_idx]["state"] == -2:
                        continue
                    cover_area = check_covered(tracking_list[z_idx]["bbox"], tracking_list[x_idx]["bbox"])
                    # print(f'area: {cover_area}')
                    if cover_area > args.covered_th:
                        print(f"z: {z_idx} is cover by x: {x_idx}")
                        tracking_list[z_idx]["be_covered"] += 1
                        z["state"] = -1
                        be_covered = True
                        break
                if be_covered and tracking_list[z_idx]["be_covered"] < args.be_covered_time_th:
                    continue
                ############
                z["no_pair_counter"] += 1
                if z["no_pair_counter"] >= args.delete_cnt and not z["is_garbage"]:  # 連續delete_cnt幀沒配到前景，且非垃圾，刪除
                    remove_idx.append(z_idx)
                    continue
    
                # 這邊的x可能是 1.本來tracking_list比較長所以沒有配到y,  2.有配到但是沒有過門檻 --> 跟前一幀的同位子比相似度
                # sim_score = get_similarity(z["bbox"], now_rgb_frame, z["bbox"], previous_rgb_frame)

                # if sim_score >= config['similarity_th']:
                #     print('no forground, no match, but still same with pre rgb frame')
                #     z["exist"] = True
                #     z["vanish_counter"] = 0
                #     z["duration"] += 1
                #     z["state"] = 1


                if z["is_garbage"] == True:
                    z["state"] = -2


            # 更新 Tracking List 各個物件的 ref 等等參數
            # 刪除
            for r_idx in remove_idx[::-1]:
                # remove_idx 中的物件必為 Tracking List 的子集合 
                del tracking_list[r_idx]

            # 更新參數
            for obj_idx, obj in enumerate(tracking_list):
                tracking_list[obj_idx]["ref"] = False

            # 設置上一幀
            previous_rgb_frame = now_rgb_frame.copy()

            # ----------Draw bbox---------- #
            for tracking_obj in tracking_list:

                if tracking_obj["state"] == 1: # 追蹤中
                    color = (0, 0, 255) # bgr
                elif tracking_obj["state"] == 0: # 剛進
                    color = (0, 255, 0)
                elif tracking_obj["state"] == -1: # 被擋住
                    color = (0 , 255, 255)
                elif tracking_obj["state"] == -2: # 垃圾
                    color = (255, 0, 0)

                tmp_bbox = tracking_obj["bbox"]
                cv2.rectangle(denoise_mask, (tmp_bbox[0], tmp_bbox[1]), (tmp_bbox[0]+tmp_bbox[2]-1, tmp_bbox[1]+tmp_bbox[3]-1), color, thickness, lineType)
                cv2.rectangle(now_rgb_frame, (tmp_bbox[0], tmp_bbox[1]), (tmp_bbox[0]+tmp_bbox[2]-1, tmp_bbox[1]+tmp_bbox[3]-1), color, thickness, lineType)

            cv2.imwrite(f"{path2output}/mask/{current_frame_idx}.png", denoise_mask)
            cv2.imwrite(f"{path2output}/rgb/{current_frame_idx}.png", now_rgb_frame)
            # ----------------------------- #
                
            print(f"Current frame: {current_frame_idx}, TrackingList Size is: {len(tracking_list)}")          

        rgb_cap.release()
        mask_cap.release()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)