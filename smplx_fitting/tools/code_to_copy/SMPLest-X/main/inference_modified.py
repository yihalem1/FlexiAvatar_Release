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
from glob import glob
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--data_format', type=str)
    args = parser.parse_args()
    assert args.root_path
    assert args.data_format in ['image', 'video']
    return args

def main():
    args = parse_args()
    cudnn.benchmark = True

    # init config
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    cur_dir = Path(__file__).resolve().parent.parent
    config_path = osp.join('./pretrained_models', 'smplest_x_h', 'config_base.py')
    cfg = Config.load_config(config_path)
    checkpoint_path = osp.join('./pretrained_models', 'smplest_x_h', 'smplest_x_h.pth.tar')

    root_path = args.root_path
    if root_path[-1] == '/':
        root_path = root_path[:-1]
    subject_id = root_path.split('/')[-1]
    img_path_list = glob(osp.join(root_path, 'frames', '*.png'))
    frame_idx_list = sorted([int(x.split('/')[-1][:-4]) for x in img_path_list])

    # (optional) used only to sanity-check image exists
    _ = cv2.imread(img_path_list[0]).shape[:2]

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

    # init human models (kept; may be used internally by Tester/model)
    smpl_x = SMPLX(cfg.model.human_model_path)

    # init tester
    demoer = Tester(cfg)
    demoer.logger.info("Using 1 GPU.")
    demoer.logger.info(f'Inference with [{cfg.model.pretrained_model_path}].')
    demoer._make_model()

    # init detector
    detector = None
    if args.data_format == 'image':
        bbox_model = getattr(cfg.inference.detection, 'model_path', './pretrained_models/yolov8x.pt')
        detector = YOLO(bbox_model)

    # IMPORTANT: match ORIGINAL saving location exactly
    # Original script saved to: <root_path>/smplx_init/<frame_idx>.json
    save_path = osp.join(root_path, 'smplx_init')
    os.makedirs(save_path, exist_ok=True)

    # (Optional) If you want ZERO extra outputs compared to original, do NOT save cam_params.
    # The original code computed focal/princpt for visualization only and did not write them.
    # So we remove cam_params writing.

    for frame_idx in tqdm(frame_idx_list):

        # prepare input image
        img_path = osp.join(root_path, 'frames', f'{frame_idx}.png')
        transform = transforms.ToTensor()
        original_img = load_img(img_path)
        original_img_height, original_img_width = original_img.shape[:2]

        # detect human bbox
        if args.data_format == 'image':
            # NOTE: your original code used FasterRCNN; here you use YOLO.
            # This does not affect the JSON *format*, only the predicted params.
            pred = detector.predict(
                original_img,
                device='cuda',
                classes=0,  # person class
                conf=cfg.inference.detection.conf,
                save=cfg.inference.detection.save,
                verbose=cfg.inference.detection.verbose
            )[0].boxes.xyxy.detach().cpu().numpy()

            if len(pred) < 1:
                continue

            bbox = pred[0].copy()           # xyxy
            bbox[2:] -= bbox[:2]            # xyxy -> xywh
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

        # crop human patch
        img, _, _ = generate_patch_image(
            cvimg=original_img,
            bbox=bbox,
            scale=1.0,
            rot=0.0,
            do_flip=False,
            out_shape=cfg.model.input_img_shape
        )
        img = transform(img.astype(np.float32)) / 255.0
        img = img.cuda()[None, :, :, :]

        inputs = {'img': img}
        targets = {}
        meta_info = {}

        # mesh recovery
        with torch.no_grad():
            out = demoer.model(inputs, targets, meta_info, 'test')

        # ------------------------------------------------------------
        # SAVE SMPL-X PARAMETERS (MATCH ORIGINAL JSON EXACTLY)
        # Keys: root_pose, body_pose, lhand_pose, rhand_pose,
        #       leye_pose, reye_pose, jaw_pose, shape, expr
        # No extra keys (e.g., no 'trans'), no missing keys.
        # Shapes/flattening match the original script exactly.
        # ------------------------------------------------------------
        root_pose  = out['smplx_root_pose'].detach().cpu().numpy()[0]
        body_pose  = out['smplx_body_pose'].detach().cpu().numpy()[0]
        lhand_pose = out['smplx_lhand_pose'].detach().cpu().numpy()[0]
        rhand_pose = out['smplx_rhand_pose'].detach().cpu().numpy()[0]
        jaw_pose   = out['smplx_jaw_pose'].detach().cpu().numpy()[0]
        shape      = out['smplx_shape'].detach().cpu().numpy()[0]
        expr       = out['smplx_expr'].detach().cpu().numpy()[0]

        out_json = {
            'root_pose': root_pose.reshape(-1).tolist(),
            'body_pose': body_pose.reshape(-1, 3).tolist(),
            'lhand_pose': lhand_pose.reshape(-1, 3).tolist(),
            'rhand_pose': rhand_pose.reshape(-1, 3).tolist(),
            'leye_pose': [0, 0, 0],
            'reye_pose': [0, 0, 0],
            'jaw_pose': jaw_pose.reshape(-1).tolist(),
            'shape': shape.reshape(-1).tolist(),
            'expr': expr.reshape(-1).tolist(),
        }

        with open(osp.join(save_path, f'{frame_idx}.json'), 'w') as f:
            json.dump(out_json, f)

if __name__ == "__main__":
    main()
