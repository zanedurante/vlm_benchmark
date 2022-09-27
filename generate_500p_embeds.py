# Generate embeddings for videos and texts from 500P data
from glob import glob
import os
import numpy as np
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm
from CLIP.CLIPVLM import ClipVLM
from video_clip import VideoClipVLM


# Change the following line to change the model used
model_name = "CLIP"

DATASET_PATH = "/home/datasets/500p/10s_dataset"
VIDEOS_DIR = "10s_clips"
TEXTS_DIR = "10s_kaldi_texts"

video_paths = sorted(glob(os.path.join(DATASET_PATH, VIDEOS_DIR) + "/*"))
text_paths = sorted(glob(os.path.join(DATASET_PATH, TEXTS_DIR) + "/*"))

# Confirm matched
print(video_paths[0:10])
print(text_paths[0:10])

# Load specified model
if model_name ==  "CLIP":
    vlm = ClipVLM(num_frames=1)
if model_name == "VideoCLIP":
    vlm = VideoClipVLM()
    vlm.load_model()

# Helper function
def get_text_from_path(text_path):
    text = ""
    with open(text_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            text += line + " "

    return text

# TODO: See if numpy vals are already stored to save time
vid_embeds = []
embed_shape = None
for video_path in tqdm(video_paths):
    try:
        vid_embed = vlm.get_video_embeds(video_path)
        if embed_shape is None:
            embed_shape = vid_embed.shape
    except:
        print("Using zero embedding!")
        vid_embed = np.zeros(embed_shape)
    vid_embeds.append(np.squeeze(vid_embed))
vid_embeds = np.asarray(vid_embeds)
print(vid_embeds.shape)
np.save('500p_10s_vid_embeds.npy', vid_embeds)

text_embeds = []
for text_path in tqdm(text_paths):
    text = get_text_from_path(text_path)
    if model_name == "VideoCLIP":
        text = [text]
    text_embed = np.squeeze(vlm.get_text_embeds(text))
    print(text_embed.shape)
    text_embeds.append(text_embed)
text_embeds = np.asarray(text_embeds)
np.save('500p_10s_text_embeds.npy', text_embeds)
print(text_embeds.shape)

# Calculate similarity with video/text
similarity = vlm.default_similarity_metric()
sim_scores = similarity(vid_embeds, text_embeds)
print(sim_scores.shape)
print(sim_scores)
y_true = list(range(len(text_embeds)))

print(y_true)

# Calculate top1, top5, top10, top20, top50 accuracy (equivalent to recall in the vid-->text retrieval scenario)
print("Video-text retrieval statistics for:", model_name)
print("Recall@1:", top_k_accuracy_score(y_true, sim_scores, k=1))
print("Recall@5:", top_k_accuracy_score(y_true, sim_scores, k=5))
print("Recall@10:", top_k_accuracy_score(y_true, sim_scores, k=10))
print("Recall@20:", top_k_accuracy_score(y_true, sim_scores, k=20))
print("Recall@50:", top_k_accuracy_score(y_true, sim_scores, k=50))
