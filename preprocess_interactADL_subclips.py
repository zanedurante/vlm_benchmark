import os, json, math, torch
import numpy as np
from tqdm.autonotebook import tqdm
from collections import defaultdict
import decord
import sys
from moviepy.editor import VideoFileClip



"""
ACTIVITY-LEVEL CLASSIFICATION
"""
INTERACTADL_DIR = "/vision/group/InteractADL_2"
TARGET_DIR = "/next/u/rharries/vlm_benchmark.data/InteractADL_egoview_activities_subclips_resized"

# Activity names which need to be changed before use
ACTIVITY_NAME_REPLACEMENTS = {
    "eat_dinner/eat_foods": "eat_food"
}

# Create directory to contain all ego-view subclips
os.makedirs(os.path.join(TARGET_DIR, "data"), exist_ok=True)

# For each separate activity, store relative video path with start/end frames appended to the end
videos_per_activity = defaultdict(list)
filenames = list(os.listdir(os.path.join(INTERACTADL_DIR, "annotations", "activity")))
if len(sys.argv) == 3:
    rank = int(sys.argv[1])
    world_size = int(sys.argv[2])
    filenames = filenames[rank::world_size]
for info_filename in tqdm(filenames):
    with open(os.path.join(INTERACTADL_DIR, "annotations", "activity", info_filename), "r") as fp:
            info = json.load(fp)

    # Collect task and person number, which identifies the overall correct mp4 video
    task_num = info["task"]
    person_num = info["person"]
    #rel_source_video_path = os.path.join("data", f"task{task_num:02}", f"Person {person_num}.mp4")
    #full_source_video_path = os.path.join(TARGET_DIR, rel_source_video_path)
    #fps = decord.VideoReader(full_source_video_path).get_avg_fps()
    source_video_path = os.path.join(INTERACTADL_DIR, "ego_view", f"task{task_num:02}", f"Person {person_num}.mp4")
    full_video = VideoFileClip(source_video_path).resize(height=256)

    for activity_count, activity_info in tqdm(list(enumerate(info["results"]))):
        activity = activity_info["activity"]
        if activity in ACTIVITY_NAME_REPLACEMENTS:
            activity = ACTIVITY_NAME_REPLACEMENTS[activity]
        activity = activity.replace("_", " ")

        activity_dir = activity.replace(" ", "_")

        # Write subclip
        relative_target_video_path = os.path.join("data", activity_dir, f"task{task_num:02}_person{person_num}_activity{activity_count:02}.mp4")
        target_video_path = os.path.join(TARGET_DIR, relative_target_video_path)
        
        if not os.path.exists(target_video_path):
            os.makedirs(os.path.dirname(target_video_path), exist_ok=True)

            start_time, end_time = activity_info["time"]
            #subclip = full_video.subclip(start_time, end_time)
            #subclip.write_videofile(target_video_path, audio=False, threads=os.cpu_count())
            from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
            ffmpeg_extract_subclip(source_video_path, start_time, end_time, target_video_path)
        
        # Ensure video is readable before including it
        try:
            vr = decord.VideoReader(target_video_path)
            
            # Save subclip path
            videos_per_activity[activity].append(relative_target_video_path)
        except RuntimeError as e:
            # Remove corrupted video
            print(relative_target_video_path, "cannot be read successfully, deleting from dataset")
            os.remove(target_video_path)

# Accumulate video information (category_dir, video_path) for each split
# Split each activity individually
train_set = {}
val_set = {}
test_set = {}
for activity, vids in videos_per_activity.items():
    train_len = round(0.6 * len(vids))
    val_len = round(0.2 * len(vids))
    test_len = len(vids) - train_len - val_len
    train_vids, val_vids, test_vids = torch.utils.data.random_split(vids, [train_len, val_len, test_len])
    train_set[activity] = list(train_vids)
    val_set[activity] = list(val_vids)
    test_set[activity] = list(test_vids)

# Save split information
os.makedirs(os.path.join(TARGET_DIR, "splits"), exist_ok=True)
with open(os.path.join(TARGET_DIR, "splits", "train.json"), "w") as fp:
    json.dump(train_set, fp, indent=4)
with open(os.path.join(TARGET_DIR, "splits", "val.json"), "w") as fp:
    json.dump(val_set, fp, indent=4)
with open(os.path.join(TARGET_DIR, "splits", "test.json"), "w") as fp:
    json.dump(test_set, fp, indent=4)
            
            

"""
ACTION-LEVEL CLASSIFICATION
"""

INTERACTADL_DIR = "/vision/group/InteractADL_2"
TARGET_DIR = "/next/u/rharries/vlm_benchmark.data/InteractADL_egoview_actions_subclips_resized"

ACTION_NAME_REPLACEMENT_RULES = {
    "/": " or ",
    "sth": "something",
    "swh": "somewhere"
}

# Create directory to contain all ego-view subclips
os.makedirs(os.path.join(TARGET_DIR, "data"), exist_ok=True)

"""
Iterate through all annotated activities, extracting corresponding segments of egoview videos and saving them into activity-labeled folders
"""
videos_per_action = defaultdict(list)
filenames = list(os.listdir(os.path.join(INTERACTADL_DIR, "annotations", "atomic_action")))
if len(sys.argv) == 3:
    rank = int(sys.argv[1])
    world_size = int(sys.argv[2])
    filenames = filenames[rank::world_size]
for info_filename in tqdm(filenames):
    with open(os.path.join(INTERACTADL_DIR, "annotations", "atomic_action", info_filename), "r") as fp:
         info = json.load(fp)
    
    # Collect task and person number, which identifies the overall correct mp4 video
    # Filename is in format: task06_person1_atomic.json
    task_num = int(info_filename[4:6])
    person_num = int(info_filename[13:14])
    source_video_path = os.path.join(INTERACTADL_DIR, "ego_view", f"task{task_num:02}", f"Person {person_num}.mp4")
    full_video = VideoFileClip(source_video_path).resize(height=256)
    
    for action_count, action_info in tqdm(list(enumerate(info))):
        action = action_info["class"]
        for k, v in ACTION_NAME_REPLACEMENT_RULES.items():
            action = action.replace(k, v)
        action_dir = action.replace(" ", "_")
        
        # Write subclip
        relative_target_video_path = os.path.join("data", action_dir, f"task{task_num:02}_person{person_num}_action{action_count:03}.webm")
        target_video_path = os.path.join(TARGET_DIR, relative_target_video_path)
        
        if not os.path.exists(target_video_path):
            os.makedirs(os.path.dirname(target_video_path), exist_ok=True)

            start_frame, end_frame = action_info["frame_start"], action_info["frame_end"]
            subclip = full_video.subclip(start_frame / 60, end_frame / 60)
            subclip.write_videofile(target_video_path, audio=False, threads=os.cpu_count())
        
        # Ensure video is readable before including it
        try:
            vr = decord.VideoReader(target_video_path)
            
            # Save subclip path
            videos_per_action[action].append(relative_target_video_path)
        except RuntimeError as e:
            # Remove corrupted video
            print(relative_target_video_path, "cannot be read successfully, deleting from dataset")
            os.remove(target_video_path)
        

# Accumulate video paths and start/end frames (category_dir, video_file) for each split
train_set = {}
val_set = {}
test_set = {}
for action, vids in videos_per_action.items():
    train_len = round(0.6 * len(vids))
    val_len = round(0.2 * len(vids))
    test_len = len(vids) - train_len - val_len
    train_vids, val_vids, test_vids = torch.utils.data.random_split(vids, [train_len, val_len, test_len])
    train_set[action] = list(train_vids)
    val_set[action] = list(val_vids)
    test_set[action] = list(test_vids)

# Save split information
os.makedirs(os.path.join(TARGET_DIR, "splits"), exist_ok=True)
with open(os.path.join(TARGET_DIR, "splits", "train.json"), "w") as fp:
    json.dump(train_set, fp, indent=4)
with open(os.path.join(TARGET_DIR, "splits", "val.json"), "w") as fp:
    json.dump(val_set, fp, indent=4)
with open(os.path.join(TARGET_DIR, "splits", "test.json"), "w") as fp:
    json.dump(test_set, fp, indent=4)