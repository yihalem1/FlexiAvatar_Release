# FlexiAvatar: Avatar Training

## 1. Environment

```bash
conda env create -f environment.yml
conda activate flexiavatar_avatar
```


```bash
# 1. PyTorch 2.0.1 + cu118
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# 2. PyTorch3D 0.7.6 
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.6"
```

Install `diff_gaussian_rasterization_depth`, a depth-returning variant of the rasterizer.
Build and install it from
[diff-gaussian-rasterization-depth](https://github.com/leo-frank/diff-gaussian-rasterization-depth),
following the instructions in that repository.

---

## 2. Model files

None of these can be redistributed. Register on each site, accept the license, and place the
files at the paths shown. Paths are relative to `common/utils/human_model_files/`.

**[SMPL-X](https://smpl-x.is.tue.mpg.de)** → `smplx/`

```
SMPLX_{NEUTRAL,MALE,FEMALE}.npz     
SMPL-X__FLAME_vertex_ids.npy        
MANO_SMPLX_vertex_ids.pkl
```

**[FLAME](https://flame.is.tue.mpg.de)** → `flame/`

```
FLAME_NEUTRAL.pkl                   
2019/generic_model.pkl              
flame_static_embedding.pkl
flame_dynamic_embedding.npy
FLAME_texture.npz
```


## 3. Directory

### Dataset

There are four loaders under `data/`. `NeuMan` reads the NeuMan release layout; the other three read the
layout Stage 1 produces. Download links for every dataset are in the
[Stage 1 README](../smplx_fitting/README.md#3-datasets). Which loader is used is chosen by the
**`DATASET` environment variable**:

```
data/
├── Custom/        Custom.py       your own videos — start here
├── NeuMan/        NeuMan.py       bike, citron, jogging, seattle
├── ZJU_MoCap/     ZJU_MoCap.py    377, 386, 387, 392, 393, 394
└── TalkShow/      TalkShow.py     oliver, conan, chemistry
```

Preprocessed NeuMan data (bike, citron, jogging, seattle), ready to place under `data/NeuMan/data/`:
[Download](https://drive.google.com/drive/folders/1Rv9lPTGmEh_T0kOWa-AfqJ0IoevrRvdX?usp=sharing).


```bash
DATASET=Custom python train.py --subject_id <subject_id>
```

### A subject folder

Put the folder Stage 1 produced at `data/<Family>/data/<subject_id>/`. It must contain:

```
data/<Family>/data/<subject_id>/
├── frames/                            <idx>.png
├── masks/                             <idx>.png
├── cam_params/                        <idx>.json
├── keypoints_whole_body/              <idx>.json
├── smplx_optimized/
│   ├── smplx_params_smoothed/         <idx>.json      per-frame pose
│   ├── shape_param.json               \
│   ├── face_offset.json                |  identity, shared across frames
│   ├── joint_offset.json               |
│   ├── locator_offset.json            /
│   ├── face_texture.png               unwrapped FLAME texture
│   └── face_texture_mask.png
├── frame_list_train.txt               training frame indices
├── frame_list_test.txt                held-out frame indices
└── bkg_point_cloud.txt                background points  (or sparse/points3D.txt from COLMAP)
```

### Generated views

To supervise regions the camera never sees, the pipeline uses diffusion-generated auxiliary views.
Producing them takes three steps:

1. **Generate the auxiliary frames.** Follow
   [MimicMotion](https://github.com/Tencent/MimicMotion) and generate roughly 500 auxiliary frames
   per subject from the captured sequence.
2. **Fit them.** Run the generated frames through
   [`smplx_fitting/`](../smplx_fitting/) exactly as in the captured video, so they come out
   in the same fitted layout (`frames/`, `masks/`, `cam_params/`, `keypoints_whole_body/`,
   `smplx_optimized/`, and the frame lists).
3. **Place the result** beside the subject as `data/<Family>/data/<subject_id>_gen/`, in the same
   layout as the subject folder above.

Without the generated data the model simply trains on captured frames only.

---

## 4. Training

```bash
conda activate flexiavatar_avatar
cd main

DATASET=Custom CUDA_VISIBLE_DEVICES=0 python train.py --subject_id <subject_id>
```

Checkpoints are written to `output/model_dump/<subject_id>/snapshot_<epoch>.pth`.

A pretrained checkpoint for the NeuMan `bike` subject is available
[Download](https://drive.google.com/drive/folders/1GvJvwoNxBVENJIorsR0FWeZAfIL8AoIe?usp=sharing).

### Animating

```bash
# drive the avatar with another sequence's SMPL-X parameters
python animate.py --subject_id <subject_id> --test_epoch 4 \
                  --motion_path ../data/Custom/data/<other_subject>

# render the canonical avatar in the neutral pose
python get_neutral_pose.py --subject_id <subject_id> --test_epoch 4
```

---

## 5. Evaluation

Following the protocol established on NeuMan by prior work [[1]](https://github.com/aipixel/GaussianAvatar/issues/14), [[2]](https://github.com/mks0601/ExAvatar_RELEASE/tree/main/avatar), evaluation optimizes the SMPL-X
parameters of the test frames against the image loss while every network parameter stays frozen. 

Four steps:

```bash
cd tools
python prepare_fit_pose_to_test.py --root_path ../output/model_dump/<subject_id>

cd ../main
# 2. optimize only the test-frame SMPL-X parameters
DATASET=NeuMan python train.py --subject_id <subject_id> --fit_pose_to_test --continue

# 3. render the test frames
DATASET=NeuMan python test.py  --subject_id <subject_id> --fit_pose_to_test --test_epoch 4

cd ../tools
python eval_neuman.py --output_path ../output/result/<subject_id>_fit_pose_to_test \
                        --subject_id <subject_id>
```

---

## Citation

```bibtex
@inproceedings{tiruneh2026flexiavatar,
  title     = {FlexiAvatar: Unified 3D Gaussian Human Avatars Under Arbitrary Body Visibility},
  author    = {Tiruneh, Yihalem Yimolal and Ali, Muhammad Salman and Jeong, Uyoung and
               Khan, Muneeb A. and Sayem, MD Khalequzzaman Chowdhury and
               Bayramgeldiyev, Allanur and Bhattarai, Binod and Baek, Seungryul},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
