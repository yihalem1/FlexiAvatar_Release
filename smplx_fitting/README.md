# FlexiAvatar — SMPL-X Fitting

## 1. Environment

```bash
conda create -y -n flexiavatar_fitting python=3.10
conda activate flexiavatar_fitting
conda install -y -c nvidia cuda-toolkit=12.4.1

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# pytorch3d 0.7.8 
pip install fvcore==0.1.5.post20221221 iopath==0.1.10
pip install --no-build-isolation --no-cache-dir \
    "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.8"

# mmcv 2.1.0 
pip install mmengine==0.10.7
git clone -b v2.1.0 --depth 1 https://github.com/open-mmlab/mmcv.git
MMCV_WITH_OPS=1 pip install --no-build-isolation -e ./mmcv

pip install -r requirements.txt

./setup.sh
pip install --no-build-isolation --no-deps -e tools/mmpose
```

---

## 2. Model files

None of these can be redistributed. Register on each site, accept the license, and place the files
at the paths shown. Paths are relative to `common/utils/human_model_files/`

**[SMPL-X](https://smpl-x.is.tue.mpg.de)** → `smplx/`

```
SMPLX_{NEUTRAL,MALE,FEMALE}.npz     SMPL-X v1.1
SMPL-X__FLAME_vertex_ids.npy
MANO_SMPLX_vertex_ids.pkl
smplx_flip_correspondences.npz
smplx_uv.zip                        extract into smplx/smplx_uv/
SMPLX_to_J14.pkl                    SMPLer-X data; Stage 3 fails without it
```

**[SMPL](https://smpl.is.tue.mpg.de)** → `smpl/`

```
SMPL_{NEUTRAL,MALE,FEMALE}.pkl      ships as basicmodel_*_lbs_10_207_0_v1.1.0.pkl — rename
```

**[FLAME](https://flame.is.tue.mpg.de)** → `flame/`

```
FLAME_NEUTRAL.pkl                   FLAME 2020 generic_model.pkl, renamed
2019/generic_model.pkl              FLAME 2019, keep this name — a different model
flame_static_embedding.pkl
flame_dynamic_embedding.npy
FLAME_texture.npz
```

**[MANO](https://mano.is.tue.mpg.de)** → `mano/`

```
MANO_{LEFT,RIGHT}.pkl
```

---

## 3. Datasets
Download each from its
official source and place it at the path shown. The loader is selected by `dataset` in
`main/config.py` (`'Custom'` by default): NeuMan uses the `NeuMan` loader, every other dataset
goes through the `Custom` loader as a plain video.

**[NeuMan](https://github.com/apple/ml-neuman)** → `data/NeuMan/data/<subject_id>/`

Preprocessed output of this stage for all four NeuMan subjects:
[Download](https://drive.google.com/drive/folders/1Rv9lPTGmEh_T0kOWa-AfqJ0IoevrRvdX?usp=sharing).

**[TalkSHOW](https://talkshow.is.tue.mpg.de)** → `data/Custom/data/<subject_id>/video.mp4`

**[ZJU-MoCap](https://chingswy.github.io/Dataset-Demo/)** → `data/Custom/data/<subject_id>/video.mp4`

---

## 4. Directory

Place your video at `data/Custom/data/<subject_id>/video.mp4`. Everything else is generated.

**Input:**

```
data/Custom/data/<subject_id>/
└── video.mp4
```

**Output:**

```
data/Custom/data/<subject_id>/
├── video.mp4
├── frames/                      extracted RGB frames, <idx>.png
├── frame_list_all.txt           frames surviving every filter stage
├── cam_params/                  per-frame camera parameters, <idx>.json
├── flame_init/                  DECA FLAME parameters
├── smplx_init/                  SMPLest-X initialisation
├── keypoints_whole_body/        mmpose 133-keypoint detections, <idx>.json
├── smplx_optimized/
│   ├── smplx_params/            per-frame fitted SMPL-X, <idx>.json
│   ├── smplx_params_smoothed/   temporally smoothed — Stage 2 reads these
│   ├── shape_param.json         identity: shape
│   ├── face_offset.json         identity: facial vertex offsets
│   ├── joint_offset.json        identity: joint offsets
│   ├── locator_offset.json      identity: joint locator offsets
│   ├── face_texture.png         unwrapped FLAME texture
│   └── face_texture_mask.png
├── masks/                       SAM foreground masks, <idx>.png
├── depthmaps_subset/            Depth-Anything-V2 depth
└── *.mp4                        per-stage visualisation videos
```

---

## 5. Fitting SMPL-X 

```bash
conda activate flexiavatar_fitting
cd tools

python extract_frames.py --root_path ../data/Custom/data/<subject_id>
python run.py            --root_path ../data/Custom/data/<subject_id>
```

Add `--use_colmap` if the the video has genuine camera motion.

---

## 6. Preparing the sequence for Stage 2

Stage 2 reads **`frame_list_train.txt`** and **`frame_list_test.txt`**

```bash
python tools/make_frame_list.py \
    --root_path data/Custom/data \
    --subjects  <subject_id>
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
