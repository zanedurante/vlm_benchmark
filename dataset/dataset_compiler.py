import os
from os import listdir
from os.path import isfile, isdir, splitext, exists
import random
import cv2
from tqdm import tqdm

'''
Given a dataset containing mp4 videos (class_folder/video.mp4), creates a frame-by-frame version of the dataset (class_folder/video_folder/frame.jpg),
then create txt files for train, val and test splits, each of which contains references to frame folders for each class, etc.
Splits follow the formats and class splits used in the FSL-Video repo.
'''

DATA_FOLDER = "D:/datasets/PAC/few_shot_act_reg"
DATASET_NAME = "kinetics_newbenchmark"
SOURCE_VIDEO_DATASET_PATH = f"{DATA_FOLDER}/{DATASET_NAME}"

TARGET_FRAME_DATASET_PATH = f"{DATA_FOLDER}/{DATASET_NAME}_frames"
TARGET_SPLIT_DATASET_PATH = f"{DATA_FOLDER}/{DATASET_NAME}_split"

IMAGE_TEMPLATE = "img_{:05d}.jpg"



def join(*paths):
    return "/".join(paths)


'''
video -> frames and split
'''
os.makedirs(TARGET_FRAME_DATASET_PATH, exist_ok=True)
os.makedirs(TARGET_SPLIT_DATASET_PATH)

cls_folder_names = [f for f in listdir(SOURCE_VIDEO_DATASET_PATH) if isdir(join(SOURCE_VIDEO_DATASET_PATH, f))]
cls_folder_names.sort()

for i, cls in enumerate(tqdm(cls_folder_names)):
    source_cls_folder_path = join(SOURCE_VIDEO_DATASET_PATH, cls)
    target_cls_folder_path = join(TARGET_FRAME_DATASET_PATH, cls)
    os.makedirs(target_cls_folder_path)
    
    if i <= 63:
        target_split_filename = "train.txt"
    elif i <= 75:
        target_split_filename = "val.txt"
    else:
        target_split_filename = "test.txt"
        
    with open(join(TARGET_SPLIT_DATASET_PATH, target_split_filename), "a") as target_split_file:    
        # Collect and shuffle all samples in class
        sample_filenames = [
            sample_filename for sample_filename in listdir(source_cls_folder_path)
            if isfile(join(source_cls_folder_path, sample_filename)) and sample_filename[0] != "."
        ]
        random.shuffle(sample_filenames)
        
        # For each sample, save frames and add reference in split file
        for sample_filename in tqdm(sample_filenames, leave=False):
            sample_video_path = join(source_cls_folder_path, sample_filename)
            
            sample_name = splitext(sample_filename)[0]
            sample_frame_folder_path = join(target_cls_folder_path, sample_name)
            
            # Save frames to folder            
            if not exists(sample_frame_folder_path):
                os.makedirs(sample_frame_folder_path)
                vidcap = cv2.VideoCapture(sample_video_path)
                framecount = 0
                success, frame = vidcap.read()
                while success:
                    framecount += 1
                    cv2.imwrite(join(sample_frame_folder_path, IMAGE_TEMPLATE.format(framecount)), frame)

                    success, frame = vidcap.read()
                
            # Save reference to sample in split txt file
            target_split_file.write(f"{sample_frame_folder_path} {framecount} {i}\n")