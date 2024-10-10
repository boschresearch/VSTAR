
import decord
from einops import rearrange
#from decord import VideoReader
from moviepy.editor import *
decord.bridge.set_bridge('torch')
from torch.utils.data.dataset import Dataset

class SingleVideoDataset(Dataset):
    def __init__(
            self,
            video_path: str,
            video_length: int = 16,
            width: int = 512,
            height: int = 320,
            video_duration: float = 2.0,
            fps: int = 8,
            tempfile_name: str = "__test_temp__.mp4"

    ):
        assert int(video_duration * fps) >= video_length
        self.video_path = video_path
        self.video_length = video_length
        self.width = width
        self.height = height
        self.video_duration = video_duration
        self.fps = fps
        self.tempfile_name = tempfile_name

    def __len__(self):
        return int(1)

    def process_video(self, old_clip):
        new_clip = old_clip.fx(vfx.speedx, old_clip.duration / self.video_duration)
        new_clip = new_clip.set_fps(self.fps)
        new_clip.write_videofile(self.tempfile_name)


    def __getitem__(self, index):
        clip = VideoFileClip(self.video_path)
        cur_duration = clip.duration
        cur_num_frames = int(clip.fps * cur_duration)
        print(f"Current video: duration={cur_duration}, FPS={clip.fps}, Frames={cur_num_frames}")
        if cur_num_frames != self.video_length:
            print(f"Processing the given video...")
            self.process_video(clip)
            read_from = self.tempfile_name
        else:
            read_from = self.video_path
        del clip

        vr = decord.VideoReader(read_from, width=self.width, height=self.height)
        start = 0
        sample_index = list(range(0, len(vr)))[start: start + self.video_length]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> c f h w")

        example = {}
        example['pixel_values'] = (video / 127.5 - 1.0)
        #example["prompt_ids"] = self.tokenize(self.prompt)

        return example