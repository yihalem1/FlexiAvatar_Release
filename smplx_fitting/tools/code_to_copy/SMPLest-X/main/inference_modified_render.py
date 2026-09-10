import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import cv2
import datetime
from tqdm import tqdm
from human_models.human_models import SMPLX
from ultralytics import YOLO
from main.base import Tester
from main.config import Config
from utils.data_utils import load_img, process_bbox, generate_patch_image
from utils.vis import render_mesh  
from glob import glob
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--data_format', type=str, required=True, choices=['image', 'video'])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cudnn.benchmark = True

    # -----------------------
    # Init config + model
    # -----------------------
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    cur_dir = Path(__file__).resolve().parent.parent
    config_path = osp.join('./pretrained_models', 'smplest_x_h', 'config_base.py')
    cfg = Config.load_config(config_path)
    checkpoint_path = osp.join('./pretrained_models', 'smplest_x_h', 'smplest_x_h.pth.tar')

    root_path = args.root_path.rstrip('/')
    subject_id = root_path.split('/')[-1]

    img_path_list = glob(osp.join(root_path, 'frames', '*.png'))
    assert len(img_path_list) > 0, f"No frames found in: {osp.join(root_path, 'frames')}"
    frame_idx_list = sorted([int(osp.basename(x)[:-4]) for x in img_path_list])

    first = cv2.imread(img_path_list[0])
    assert first is not None, f"Failed to read first frame: {img_path_list[0]}"
    img_height, img_width = first.shape[:2]

    exp_name = f'inference_{subject_id}_{time_str}'
    new_config = {
        "model": {
            "pretrained_model_path": checkpoint_path,
        },
        "log": {
            'exp_name': exp_name,
            'log_dir': osp.join(cur_dir, 'outputs', exp_name, 'log'),
        }
    }
    cfg.update_config(new_config)
    cfg.prepare_log()

    # init SMPL-X model wrapper (kept; used for faces)
    smpl_x = SMPLX(cfg.model.human_model_path)

    # init tester + load network
    demoer = Tester(cfg)
    demoer.logger.info("Using 1 GPU.")
    demoer.logger.info(f'Inference with [{cfg.model.pretrained_model_path}].')
    demoer._make_model()

    # -----------------------
    # Init detector
    # -----------------------
    detector = None
    if args.data_format == 'image':
        bbox_model = getattr(cfg.inference.detection, 'model_path', './pretrained_models/yolov8x.pt')
        detector = YOLO(bbox_model)

    # -----------------------
    # Output paths (match original style)
    # -----------------------
    # 1) smplx params folder exactly like original
    smplx_save_dir = osp.join(root_path, 'smplx_init')
    os.makedirs(smplx_save_dir, exist_ok=True)

    # 2) side-by-side video exactly like original
    video_out_path = osp.join(root_path, 'smplx_init.mp4')
    fps = 30
    video_save = cv2.VideoWriter(
        video_out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (img_width * 2, img_height)  # side-by-side output
    )

    # get faces (handle different attribute naming)
    faces = getattr(smpl_x, 'face', None)
    if faces is None:
        faces = getattr(smpl_x, 'faces', None)
    if faces is None:
        raise AttributeError("SMPLX wrapper has no 'face' or 'faces' attribute (needed for render_mesh).")

    # -----------------------
    # Main loop
    # -----------------------
    for frame_idx in tqdm(frame_idx_list):

        img_path = osp.join(root_path, 'frames', f'{frame_idx}.png')
        transform = transforms.ToTensor()

        # load original image (likely RGB depending on your loader)
        original_img = load_img(img_path)
        if original_img is None:
            continue

        original_img_height, original_img_width = original_img.shape[:2]

        # -----------------------
        # Detect bbox
        # -----------------------
        if args.data_format == 'image':
            pred = detector.predict(
                original_img,
                device='cuda',
                classes=0,  # person
                conf=cfg.inference.detection.conf,
                save=cfg.inference.detection.save,
                verbose=cfg.inference.detection.verbose
            )[0].boxes.xyxy.detach().cpu().numpy()

            if len(pred) < 1:
                continue

            bbox = pred[0].copy()     # xyxy
            bbox[2:] -= bbox[:2]      # -> xywh
        else:
            bbox_path = osp.join(root_path, 'bboxes', f'{frame_idx}.json')
            if not osp.isfile(bbox_path):
                continue
            with open(bbox_path) as f:
                bbox = np.array(json.load(f), dtype=np.float32)  # xywh

        bbox = process_bbox(
            bbox=bbox,
            img_width=original_img_width,
            img_height=original_img_height,
            input_img_shape=cfg.model.input_img_shape,
            ratio=getattr(cfg.data, "bbox_ratio", 1.25)
        )
        if bbox is None:
            continue

        # -----------------------
        # Crop human patch
        # -----------------------
        img_patch, _, _ = generate_patch_image(
            cvimg=original_img,
            bbox=bbox,
            scale=1.0,
            rot=0.0,
            do_flip=False,
            out_shape=cfg.model.input_img_shape
        )
        img_patch = transform(img_patch.astype(np.float32)) / 255.0
        img_patch = img_patch.cuda()[None, :, :, :]

        inputs = {'img': img_patch}
        targets = {}
        meta_info = {}

        # -----------------------
        # Inference
        # -----------------------
        with torch.no_grad():
            out = demoer.model(inputs, targets, meta_info, 'test')

        # -----------------------
        # Render video frame (same as original)
        # -----------------------
        mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]  # (V,3)

        # Convert to BGR for cv2 output (original code wrote BGR)
        vis_img_bgr = original_img[:, :, ::-1].copy()

        # Compute focal/princpt exactly like original (but using cfg.model.*)
        focal = [
            cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2],
            cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3],
        ]
        princpt = [
            cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0],
            cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1],
        ]

        rendered_img = render_mesh(
            vis_img_bgr.copy(),
            mesh,
            faces,
            {'focal': focal, 'princpt': princpt}
        )

        # side-by-side frame + index text
        frame = np.concatenate((vis_img_bgr, rendered_img), axis=1)

        # ensure output size matches VideoWriter
        if frame.shape[0] != img_height or frame.shape[1] != img_width * 2:
            frame = cv2.resize(frame, (img_width * 2, img_height), interpolation=cv2.INTER_LINEAR)

        frame = cv2.putText(
            frame,
            str(frame_idx),
            (int(img_width * 0.1), int(img_height * 0.1)),
            cv2.FONT_HERSHEY_PLAIN,
            2.0,
            (0, 0, 255),
            3
        )
        video_save.write(frame.astype(np.uint8))

        # -----------------------
        # Save SMPL-X JSON (EXACTLY like your original)
        # -----------------------
        root_pose = out['smplx_root_pose'].detach().cpu().numpy()[0]
        body_pose = out['smplx_body_pose'].detach().cpu().numpy()[0]
        lhand_pose = out['smplx_lhand_pose'].detach().cpu().numpy()[0]
        rhand_pose = out['smplx_rhand_pose'].detach().cpu().numpy()[0]
        jaw_pose = out['smplx_jaw_pose'].detach().cpu().numpy()[0]
        shape = out['smplx_shape'].detach().cpu().numpy()[0]
        expr = out['smplx_expr'].detach().cpu().numpy()[0]

        with open(osp.join(smplx_save_dir, f'{frame_idx}.json'), 'w') as f:
            json.dump({
                'root_pose': root_pose.reshape(-1).tolist(),
                'body_pose': body_pose.reshape(-1, 3).tolist(),
                'lhand_pose': lhand_pose.reshape(-1, 3).tolist(),
                'rhand_pose': rhand_pose.reshape(-1, 3).tolist(),
                'leye_pose': [0, 0, 0],
                'reye_pose': [0, 0, 0],
                'jaw_pose': jaw_pose.reshape(-1).tolist(),
                'shape': shape.reshape(-1).tolist(),
                'expr': expr.reshape(-1).tolist()
            }, f)

    video_save.release()
    print(f"[Done] Saved video: {video_out_path}")
    print(f"[Done] Saved SMPL-X params: {smplx_save_dir}")


if __name__ == "__main__":
    main()
