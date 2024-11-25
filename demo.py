import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from tqdm import tqdm

from config import config as cfg
from test_tools.common import detect_all, grab_all_frames
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.supply_writer import SupplyWriter
from test_tools.utils import get_crop_box
from utils.plugin_loader import PluginLoader

# Constants for normalization
mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1, 1)
std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1, 1)
max_frame = 400
video_directory = "dataset/ff_manipulated"  # Directory containing all video files
out_dir = "prediction"
cfg_path = "i3d_ori.yaml"
ckpt_path = "checkpoints/model.pth"
optimal_threshold = 0.04

def save_scores_to_excel(video_name, avg_score, output_path):
    """
    Save the video name and score to an Excel file.

    Parameters:
    video_name (str): The name of the video without the .mp4 extension.
    avg_score (float): The average score of the video.
    output_path (str): Path to save the Excel file.
    """
    new_data = pd.DataFrame({"Video Name": [video_name], "Score": [round(avg_score, 4)]})
    
    if os.path.exists(output_path):
        # Read existing data
        existing_data = pd.read_excel(output_path)
        # Append new data
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        # Create new data
        updated_data = new_data
    
    # Save the updated data back to the file
    updated_data.to_excel(output_path, index=False, sheet_name="Scores")



if __name__ == "__main__":
    cfg.init_with_yaml()
    cfg.update_with_yaml(cfg_path)
    cfg.freeze()

    classifier = PluginLoader.get_classifier(cfg.classifier_type)()
    classifier.cuda()
    classifier.eval()
    classifier.load(ckpt_path)

    crop_align_func = FasterCropAlignXRay(cfg.imsize)
    os.makedirs(out_dir, exist_ok=True)
    #add file path
    excel_output_path = os.path.join(out_dir, "video_scores_altfreeze.xlsx")

    for video_file in os.listdir(video_directory):
        if video_file.endswith(".mp4"):
            torch.cuda.empty_cache()
            print(f"Processing video: {video_file}")
            video_path = os.path.join(video_directory, video_file)
            basename = f"{os.path.splitext(os.path.basename(video_path))[0]}.avi"
            out_file = os.path.join(out_dir, basename)
            cache_file = f"{video_path}_{max_frame}.pth"

            if os.path.exists(cache_file):
                detect_res, all_lm68 = torch.load(cache_file)
                frames = grab_all_frames(video_path, max_size=max_frame, cvt=True)
                # print("Detection result loaded from cache")
            else:
                # print("Detecting...")
                try:
                    detect_res, all_lm68, frames = detect_all(video_path, return_frames=True, max_size=max_frame)
                except Exception as e:
                    print(f"Error grabbing frames: {e}")
                    continue
                torch.save((detect_res, all_lm68), cache_file)
                # print("Detection finished")

            # print(f"Number of frames: {len(frames)}")
            shape = frames[0].shape[:2]
            all_detect_res = []

            assert len(all_lm68) == len(detect_res)

            for faces, faces_lm68 in zip(detect_res, all_lm68):
                new_faces = []
                for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
                    new_face = (box, lm5, face_lm68, score)
                    new_faces.append(new_face)
                all_detect_res.append(new_faces)

            detect_res = all_detect_res
            # print("Split into super clips")

            tracks = multiple_tracking(detect_res)
            tuples = [(0, len(detect_res))] * len(tracks)

            # print("Full tracks:", len(tracks))

            if len(tracks) == 0:
                tuples, tracks = find_longest(detect_res)

            data_storage = {}
            frame_boxes = {}
            super_clips = []

            for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)):
                # print(f"Track {track_i}: Start {start}, End {end}")
                assert len(detect_res[start:end]) == len(track)

                super_clips.append(len(track))

                for face, frame_idx, j in zip(track, range(start, end), range(len(track))):
                    box, lm5, lm68 = face[:3]
                    big_box = get_crop_box(shape, box, scale=0.5)

                    top_left = big_box[:2][None, :]
                    new_lm5 = lm5 - top_left
                    new_lm68 = lm68 - top_left
                    new_box = (box.reshape(2, 2) - top_left).reshape(-1)
                    info = (new_box, new_lm5, new_lm68, big_box)

                    x1, y1, x2, y2 = big_box
                    cropped = frames[frame_idx][y1:y2, x1:x2]
                    base_key = f"{track_i}_{j}_"
                    data_storage[f"{base_key}img"] = cropped
                    data_storage[f"{base_key}ldm"] = info
                    data_storage[f"{base_key}idx"] = frame_idx
                    frame_boxes[frame_idx] = np.rint(box).astype(int)

            # print("Sampling clips from super clips:", super_clips)

            clips_for_video = []
            clip_size = cfg.clip_size
            pad_length = clip_size - 1

            for super_clip_idx, super_clip_size in enumerate(super_clips):
                inner_index = list(range(super_clip_size))
                if super_clip_size < clip_size:  # padding
                    post_module = inner_index[1:-1][::-1] + inner_index

                    l_post = len(post_module)
                    post_module = post_module * (pad_length // l_post + 1)
                    post_module = post_module[:pad_length]
                    assert len(post_module) == pad_length

                    pre_module = inner_index + inner_index[1:-1][::-1]
                    l_pre = len(pre_module)
                    pre_module = pre_module * (pad_length // l_pre + 1)
                    pre_module = pre_module[-pad_length:]
                    assert len(pre_module) == pad_length

                    inner_index = pre_module + inner_index + post_module

                super_clip_size = len(inner_index)

                frame_range = [
                    inner_index[i : i + clip_size] for i in range(super_clip_size) if i + clip_size <= super_clip_size
                ]
                for indices in frame_range:
                    clip = [(super_clip_idx, t) for t in indices]
                    clips_for_video.append(clip)

            preds = []
            frame_res = {}

            for clip in tqdm(clips_for_video, desc="Testing clips"):
                images = [data_storage[f"{i}_{j}_img"] for i, j in clip]
                landmarks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip]
                frame_ids = [data_storage[f"{i}_{j}_idx"] for i, j in clip]
                _, images_align = crop_align_func(landmarks, images)
                for i in range(clip_size):
                    img1 = cv2.resize(images[i], (cfg.imsize, cfg.imsize))
                    img = np.concatenate((img1, images_align[i]), axis=1)
                images = torch.as_tensor(images_align, dtype=torch.float32).cuda().permute(3, 0, 1, 2)
                images = images.unsqueeze(0).sub(mean).div(std)

                with torch.no_grad():
                    output = classifier(images)
                pred = float(F.sigmoid(output["final_output"]))
                for f_id in frame_ids:
                    if f_id not in frame_res:
                        frame_res[f_id] = []
                    frame_res[f_id].append(pred)
                preds.append(pred)
            avg_score = np.mean(preds)
            print(avg_score)

            boxes = []
            scores = []

            for frame_idx in range(len(frames)):
                if frame_idx in frame_res:
                    pred_prob = np.mean(frame_res[frame_idx])
                    rect = frame_boxes[frame_idx]
                else:
                    pred_prob = None
                    rect = None
                scores.append(pred_prob)
                boxes.append(rect)
            save_scores_to_excel(
                video_name=os.path.splitext(video_file)[0],
                avg_score=avg_score,
                output_path=excel_output_path
            )
            SupplyWriter(video_path, out_file, optimal_threshold).run(frames, scores, boxes)
