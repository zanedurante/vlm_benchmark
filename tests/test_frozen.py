import unittest
import os
import numpy as np

from pytube import YouTube

from frozen.frozen_vlm import FrozenVLM
from similarity_metrics import Similarity

VIDEO_DIR_PATH = "./test_videos/"
VIDEO_LABELS = ["Cat", "Basketball", "Dog"]


class FrozenSimpleTest(unittest.TestCase):
    def test_frozen(self):
        # Instantiate VideoCLIP
        vlm = FrozenVLM()
        text_embeds = np.asarray([vlm.get_text_embeds(label) for label in VIDEO_LABELS])
        video_paths = [os.path.join(VIDEO_DIR_PATH, lab+".mp4") for lab in VIDEO_LABELS]
        video_embeds = []
        for path in video_paths:
            video_embeds.append(vlm.get_video_embeds(path).squeeze())
        video_embeds = np.asarray(video_embeds)
        outputs = Similarity.DOT(video_embeds, text_embeds)
        preds = np.argmax(outputs, -1)
        print(outputs)
        assert np.array_equal(preds, [0, 1, 2])

# TODO: Move to test utils files
def download_videos():
    print("Downloading test videos...")

    video_paths = [
        "https://www.youtube.com/watch?v=-5CdAup0o-I",
        "https://www.youtube.com/watch?v=ue1NT3QhuVU",
        "https://www.youtube.com/watch?v=28xjtYY3V3Q",
    ]

    for name, link in zip(VIDEO_LABELS, video_paths):
        url = YouTube(link)
        print("downloading....")
        video = url.streams.get_highest_resolution()
        video.download(output_path=VIDEO_DIR_PATH, filename=name+".mp4")
        print("Downloaded:", name)


if __name__ == '__main__':
    # Load videos to test CLIP
    videos_loaded = True  # Change to True if videos are already loaded to save time
    if not videos_loaded:
        download_videos()

    unittest.main()
