#!/usr/bin/env bash


set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$ROOT_DIR/tools"

info()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn()  { printf '\033[1;33m    %s\033[0m\n' "$*"; }

clone() {
    local url="$1" dir="$2" ref="$3"
    if [ -d "$TOOLS_DIR/$dir/.git" ]; then
        echo "    $dir already cloned, skipping"
        return
    fi
    echo "    cloning $dir @ $ref"
    git clone --quiet "$url" "$TOOLS_DIR/$dir"
    git -C "$TOOLS_DIR/$dir" checkout --quiet "$ref"
}

fetch() {
    local url="$1" dst="$2"
    if [ -s "$dst" ]; then
        echo "    $(basename "$dst") already present, skipping"
        return
    fi
    mkdir -p "$(dirname "$dst")"
    echo "    downloading $(basename "$dst")"
    wget --quiet --show-progress -O "$dst.part" "$url"
    mv "$dst.part" "$dst"
}

# ---------------------------------------------------------------------------
info "1/4  Cloning upstream repositories"
# ---------------------------------------------------------------------------

clone https://github.com/open-mmlab/mmpose.git          mmpose            v1.3.2
clone https://github.com/facebookresearch/segment-anything.git segment-anything main
clone https://github.com/DepthAnything/Depth-Anything-V2.git   Depth-Anything-V2 main
clone https://github.com/yfeng95/DECA.git               DECA              master
clone https://github.com/SMPLCap/SMPLest-X.git          SMPLest-X         main

# ---------------------------------------------------------------------------
info "2/4  Applying modifications (tools/copy_code.py)"
# ---------------------------------------------------------------------------
( cd "$TOOLS_DIR" && python copy_code.py )


if [ ! -e "$TOOLS_DIR/SMPLest-X/human_models/human_model_files" ]; then
    mkdir -p "$TOOLS_DIR/SMPLest-X/human_models"
    ln -s ../../../common/utils/human_model_files \
          "$TOOLS_DIR/SMPLest-X/human_models/human_model_files"
    echo "    linked SMPLest-X/human_models/human_model_files"
fi

# ---------------------------------------------------------------------------
info "3/4  Downloading openly-licensed weights"
# ---------------------------------------------------------------------------
fetch https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
      "$TOOLS_DIR/segment-anything/sam_vit_h_4b8939.pth"

fetch https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth \
      "$TOOLS_DIR/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth"

fetch https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth \
      "$TOOLS_DIR/mmpose/dw-ll_ucoco_384.pth"

fetch https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
      "$TOOLS_DIR/mmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"

fetch https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt \
      "$TOOLS_DIR/SMPLest-X/pretrained_models/yolov8x.pt"
info "Done."
