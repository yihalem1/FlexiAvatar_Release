import os
import os.path as osp
import shutil
import sys

TOOLS_DIR = osp.dirname(osp.abspath(__file__))
SRC_DIR = osp.join(TOOLS_DIR, 'code_to_copy')

FILES = [
    ('run_colmap.py', 'COLMAP/run_colmap.py'),

    # DECA
    ('run_deca.py', 'DECA/run_deca.py'),
    ('DECA/decalib/deca.py', 'DECA/decalib/deca.py'),
    ('DECA/decalib/datasets/datasets.py', 'DECA/decalib/datasets/datasets.py'),
    ('DECA/demos/demo_reconstruct.py', 'DECA/demos/demo_reconstruct.py'),

    # mmpose
    ('run_mmpose.py', 'mmpose/run_mmpose.py'),
    ('run_mmpose_lower_body_fixed.py', 'mmpose/run_mmpose_lower_body_fixed.py'),
    ('mmpose/demo/topdown_demo_with_mmdet.py', 'mmpose/demo/topdown_demo_with_mmdet.py'),

    # segment-anything
    ('run_sam.py', 'segment-anything/run_sam.py'),

    # Depth-Anything-V2
    ('run_depth_anything.py', 'Depth-Anything-V2/run_depth_anything.py'),

    # SMPLest-X - run.py stage 3 invokes main/inference_modified_render.py
    ('SMPLest-X/main/inference_modified_render.py', 'SMPLest-X/main/inference_modified_render.py'),
    ('SMPLest-X/main/inference_modified.py', 'SMPLest-X/main/inference_modified.py'),
    ('SMPLest-X/main/inference.py', 'SMPLest-X/main/inference.py'),
    ('SMPLest-X/utils/vis.py', 'SMPLest-X/utils/vis.py'),
]

REQUIRED_REPOS = ['DECA', 'mmpose', 'segment-anything', 'Depth-Anything-V2', 'SMPLest-X']


def main():
    missing_repos = [r for r in REQUIRED_REPOS if not osp.isdir(osp.join(TOOLS_DIR, r))]
    if missing_repos:
        print('ERROR: these repos have not been cloned yet: ' + ', '.join(missing_repos))
        print('Run ./setup.sh first.')
        return 1

    errors = []
    for src_rel, dst_rel in FILES:
        src = osp.join(SRC_DIR, src_rel)
        dst = osp.join(TOOLS_DIR, dst_rel)

        if not osp.isfile(src):
            errors.append('missing source: code_to_copy/' + src_rel)
            continue

        os.makedirs(osp.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        print('  ' + dst_rel)

    if errors:
        print('\nERROR:')
        for e in errors:
            print('  ' + e)
        return 1

    print('\nCopied %d files.' % len(FILES))
    return 0


if __name__ == '__main__':
    sys.exit(main())
