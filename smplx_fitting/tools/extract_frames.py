import os
import os.path as osp
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
root_path = args.root_path
os.makedirs(osp.join(root_path, 'frames'), exist_ok=True)

vid_path = osp.join(root_path, 'video.mp4')
vidcap = cv2.VideoCapture(vid_path)
if not vidcap.isOpened():
    raise RuntimeError(f"Failed to open video: {vid_path}")

frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
success, frame = vidcap.read()
frame_idx = 0

list_all = osp.join(root_path, 'frame_list_all.txt')
list_train = osp.join(root_path, 'frame_list_train.txt')

with open(list_all, 'w', encoding='utf-8') as f_all, \
     open(list_train, 'w', encoding='utf-8') as f_train:

    while success and frame_idx < frame_num:  
        print(f"{frame_idx}/{frame_num}", end='\r')
        cv2.imwrite(osp.join(root_path, 'frames', f"{frame_idx}.png"), frame)
        
        f_all.write(f"{frame_idx}\n")
        f_train.write(f"{frame_idx}\n")

        success, frame = vidcap.read()
        frame_idx += 1

vidcap.release()
print(f"\nDone. Saved {frame_idx} frames (total {frame_num}) and wrote indices to both lists.")
