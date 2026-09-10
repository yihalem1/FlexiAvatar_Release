import os
import os.path as osp
from glob import glob
import sys
import argparse
import shutil
import subprocess
import time
from contextlib import contextmanager

TOOLS_DIR = osp.dirname(osp.abspath(__file__))
ROOT_DIR = osp.dirname(TOOLS_DIR)
MAIN_DIR = osp.join(ROOT_DIR, 'main')
OUTPUT_DIR = osp.join(ROOT_DIR, 'output')


_STAGE_TIMES = []  # list of (stage_name, seconds)

def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.2f}min"
    hours = minutes / 60.0
    return f"{hours:.2f}hr"

@contextmanager
def timed_stage(stage_name: str):
    t0 = time.perf_counter()
    print(f"\n==================== {stage_name} (START) ====================")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        _STAGE_TIMES.append((stage_name, dt))
        print(f"==================== {stage_name} (END) ======================")
        print(f"[TIME] {stage_name}: {_fmt_time(dt)}\n")

def print_timing_breakdown():
    print("\n==================== TIMING BREAKDOWN ====================")
    total = 0.0
    for name, sec in _STAGE_TIMES:
        total += sec
        print(f"- {name:<35} {_fmt_time(sec):>10}")
    print("----------------------------------------------------------")
    # print(f"TOTAL: {_fmt_time(total)}")
    print("==========================================================\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--use_colmap', dest='use_colmap', action='store_true')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
 
    args.root_path = osp.abspath(args.root_path)
    return args

def load_frame_list(root_path, frame_list_name='frame_list_all.txt'):
    frame_list_file = osp.join(root_path, frame_list_name)
    with open(frame_list_file) as f:
        frame_idx_list = [int(x.strip()) for x in f.readlines()]
    return frame_idx_list

def remove_unnecessary_frames(root_path, frame_idx_list):

    img_path_list = glob(osp.join(root_path, 'frames', '*.png')) + \
                    glob(osp.join(root_path, 'frames', '*.jpg'))
    for img_path in img_path_list:
        frame_idx = int(osp.splitext(osp.basename(img_path))[0])
        if frame_idx not in frame_idx_list:
            cmd = 'rm ' + img_path
            result = os.system(cmd)
            if result != 0:
                print('Something bad happened when removing unnecessary frames. Terminate the script.')
                sys.exit()
    print("Removed unnecessary frames.")

def remove_folder(target_folder):

    if osp.exists(target_folder):
        print(f"Removing existing folder: {target_folder}")
        shutil.rmtree(target_folder)
    os.makedirs(target_folder, exist_ok=True)
    print(f"Created new folder: {target_folder}")

def filter_frames(root_path, frame_idx_list, rel_folder, ext='.json'):
 
    valid_frames = []
    for idx in frame_idx_list:
        file_path = osp.join(root_path, rel_folder, f"{idx}{ext}")
        if osp.exists(file_path):
            valid_frames.append(idx)
        else:
            print(f"Frame {idx} is missing expected file: {file_path}. It will be excluded.")
    return valid_frames

def update_frame_list_file(root_path, frame_idx_list, frame_list_name='frame_list_all.txt'):

    file_path = osp.join(root_path, frame_list_name)
    with open(file_path, 'w') as f:
        for idx in frame_idx_list:
            f.write(f"{idx}\n")
    print(f"Frame list updated: {file_path}")
    return file_path

# ---------------------------
# Main pipeline start
# ---------------------------

# total pipeline timer
_total_t0 = time.perf_counter()

try:
    with timed_stage("Parse args + setup"):
        args = parse_args()
        root_path = args.root_path
        if root_path[-1] == '/':
            subject_id = root_path.split('/')[-2]
        else:
            subject_id = root_path.split('/')[-1]

    with timed_stage("Load frame list + remove unnecessary frames + update list"):
        frame_idx_list = load_frame_list(root_path, 'frame_list_all.txt')
        remove_unnecessary_frames(root_path, frame_idx_list)
        update_frame_list_file(root_path, frame_idx_list, 'frame_list_all.txt')

    # #Stage 1: Create Camera Parameters
    if args.use_colmap:
        with timed_stage("Stage 1: COLMAP camera params"):
            sparse_folder = osp.join(root_path, 'sparse')
            remove_folder(sparse_folder)
            os.chdir(osp.join(TOOLS_DIR, 'COLMAP'))
            cmd = 'python run_colmap.py --root_path ' + root_path
            print(cmd)
            result = os.system(cmd)
            if result != 0:
                print('Something bad happened when running COLMAP. Terminate the script.')
                sys.exit()
            os.chdir(TOOLS_DIR)
    else:
        with timed_stage("Stage 1: Virtual camera params"):
            cam_params_folder = osp.join(root_path, 'cam_params')
            remove_folder(cam_params_folder)
            os.chdir(TOOLS_DIR)
            cmd = 'python make_virtual_cam_params.py --root_path ' + root_path
            print(cmd)
            result = os.system(cmd)
            if result != 0:
                print('Something bad happened when making the virtual camera parameters. Terminate the script.')
                sys.exit()

    # ---------------------------
    # Stage 2: DECA (FLAME Parameters)
    # ---------------------------
    with timed_stage("Stage 2: DECA (FLAME params)"):
        flame_params_folder = osp.join(root_path, 'flame_init', 'flame_params')
        remove_folder(flame_params_folder)
        os.chdir(osp.join(TOOLS_DIR, 'DECA'))
        cmd = 'python run_deca.py --root_path ' + root_path
        print(cmd)
        result = os.system(cmd)
        if result != 0:
            print('Something bad happened when running DECA. Terminate the script.')
            sys.exit()
        os.chdir(TOOLS_DIR)

    with timed_stage("Filter frames by FLAME params + update list"):
        frame_idx_list = filter_frames(root_path, frame_idx_list, osp.join('flame_init', 'flame_params'))
        update_frame_list_file(root_path, frame_idx_list, 'frame_list_all.txt')

    # ---------------------------
    # Stage 3: SMPLest-X (initial SMPLX Parameters)
    # ---------------------------
    with timed_stage("Stage 3: SMPLest-X (SMPLX init)"):

        os.chdir(osp.join(TOOLS_DIR, 'SMPLest-X'))
        cmd = 'python main/inference_modified_render.py --root_path ' + root_path + ' --data_format image'
        print(cmd)
        result = os.system(cmd)
        if result != 0:
            print('Something bad happened when running SMPLest-X. Terminate the script.')
            sys.exit()
        os.chdir(TOOLS_DIR)

    with timed_stage("Filter frames by SMPLX params + update list"):
        # Filter frames by checking the existence of SMPLX parameter files.
        frame_idx_list = filter_frames(root_path, frame_idx_list, 'smplx_init')
        update_frame_list_file(root_path, frame_idx_list, 'frame_list_all.txt')

    # ---------------------------
    # Stage 4: mmpose (2D Keypoints)
    # ---------------------------
    with timed_stage("Stage 4: mmpose"):
        keypoints_folder = osp.join(root_path, 'keypoints_whole_body')
        remove_folder(keypoints_folder)
        os.chdir(osp.join(TOOLS_DIR, 'mmpose'))
        cmd = 'python run_mmpose.py --root_path ' + root_path
        print(cmd)
        result = os.system(cmd)
        if result != 0:
            print('Something bad happened when running mmpose. Terminate the script.')
            sys.exit()
        os.chdir(TOOLS_DIR)

    with timed_stage("Filter frames by keypoints + update list"):
        frame_idx_list = filter_frames(root_path, frame_idx_list, 'keypoints_whole_body')
        update_frame_list_file(root_path, frame_idx_list, 'frame_list_all.txt')

    # ---------------------------
    # Stage 5: Fit SMPLX
    # ---------------------------
    with timed_stage("Stage 5: Fit SMPLX (fit.py + move outputs)"):
        os.chdir(MAIN_DIR)
        cmd = 'python fit.py --subject_id ' + subject_id
        print(cmd)
        result = os.system(cmd)
        if result != 0:
            print('Something bad happened when fitting. Terminate the script.')
            sys.exit()
        os.chdir(TOOLS_DIR)
        cmd = 'mv ' + osp.join(OUTPUT_DIR, 'result', subject_id, '*') + ' ' + osp.join(root_path, '.')
        print(cmd)
        result = os.system(cmd)
        if result != 0:
            print('Something bad happened when moving the fitted files to root_path. Terminate the script.')
            sys.exit()

    # ---------------------------
    # Stage 6: Unwrap Textures for FLAME
    # ---------------------------
    with timed_stage("Stage 6: Unwrap FLAME textures (unwrap.py + move outputs)"):
        smplx_optimized_folder = osp.join(root_path, 'smplx_optimized')
        os.chdir(MAIN_DIR)
        cmd = 'python unwrap.py --subject_id ' + subject_id
        print(cmd)
        result = os.system(cmd)
        if result != 0:
            print('Something bad happened when unwrapping the face images to FLAME UV texture. Terminate the script.')
            sys.exit()
        os.chdir(TOOLS_DIR)
        cmd = 'mv ' + osp.join(OUTPUT_DIR, 'result', subject_id, 'unwrapped_textures', '*') + ' ' + smplx_optimized_folder
        print(cmd)
        result = os.system(cmd)
        if result != 0:
            print('Something bad happened when moving the unwrapped FLAME UV texture to root_path. Terminate the script.')
            sys.exit()

    # ---------------------------
    # Stage 7: Smooth SMPLX Parameters
    # ---------------------------
    with timed_stage("Stage 7: Smooth SMPLX params"):
        os.chdir(TOOLS_DIR)
        cmd = 'python smooth_smplx_params.py --root_path ' + root_path
        print(cmd)
        result = os.system(cmd)
        if result != 0:
            print('Something bad happened when smoothing smplx parameters. Terminate the script.')
            sys.exit()

    # ---------------------------
    # Stage 8: Segmentation Mask (Segment Anything)
    # ---------------------------
    with timed_stage("Stage 8: Segmentation (SAM)"):
        os.chdir(osp.join(TOOLS_DIR, 'segment-anything'))
        cmd = 'python run_sam.py --root_path ' + root_path
        print(cmd)
        result = os.system(cmd)
        if result != 0:
            print('Something bad happened when running the segmentation. Terminate the script.')
            sys.exit()
        os.chdir(TOOLS_DIR)

    # ---------------------------
    # Stage 9: Depth Estimation (Depth Anything V2)
    # ---------------------------
    with timed_stage("Stage 9: Depth Anything V2"):
        os.chdir(osp.join(TOOLS_DIR, 'Depth-Anything-V2'))
        cmd = 'python run_depth_anything.py --root_path ' + root_path
        print(cmd)
        result = os.system(cmd)
        if result != 0:
            print('Something bad happened when running depth estimation. Terminate the script.')
            sys.exit()
        os.chdir(TOOLS_DIR)

    print("All processing stages completed successfully!")

finally:
    total_dt = time.perf_counter() - _total_t0
    _STAGE_TIMES.append(("TOTAL PIPELINE", total_dt))
    print_timing_breakdown()
