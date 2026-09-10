import cv2
import torch
import os
import os.path as osp
from glob import glob
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, dest='output_path')
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--include_bkg', dest='include_bkg', action='store_true')
    parser.add_argument('--dataset', type=str, dest='dataset', default=None,
                        help='dataset folder under data/ (default: $DATASET, else Custom)')
    args = parser.parse_args()
    assert args.output_path, "Please set output_path."
    assert args.subject_id, "Please set subject_id."
    return args

args = parse_args()
output_path = args.output_path
subject_id = args.subject_id
include_bkg = args.include_bkg


dataset = args.dataset or os.environ.get('DATASET', 'ZJU_MoCap')  # Custom, NeuMan
data_root = osp.join('..', 'data', dataset, 'data', subject_id)
assert osp.isdir(data_root), \
    "%s not found. Set --dataset or $DATASET to the family holding '%s'." % (data_root, subject_id)

# Initialize metrics.
results = {'psnr': [], 'ssim': [], 'lpips': []}
psnr = PeakSignalNoiseRatio(data_range=1).cuda()
ssim = StructuralSimilarityIndexMeasure(data_range=1).cuda()
lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").cuda()

images_folder = "images/" + subject_id
if not os.path.exists(images_folder):
    os.makedirs(images_folder)

# Read the frame indices.
with open(osp.join(data_root, 'frame_list_test.txt')) as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    frame_idx = int(line)

    # Load output image.
    out_path = osp.join(output_path, f"{frame_idx}_scene_human_refined_composed.png")
    out = cv2.imread(out_path)[:, :, ::-1] / 255.
    out = torch.FloatTensor(out).permute(2, 0, 1)[None, :, :, :].cuda()

    # Load ground truth image.
    gt_path = osp.join(data_root, 'frames', f'{frame_idx}.png')
    gt = cv2.imread(gt_path)[:, :, ::-1] / 255.
    gt = torch.FloatTensor(gt).permute(2, 0, 1)[None, :, :, :].cuda()

    # Load ground truth mask.
    mask_path = osp.join(data_root, 'masks', f'{frame_idx}.png')
    mask = cv2.imread(mask_path)
    mask = mask / 255.0  # Now 1 = person, 0 = background
    mask = torch.FloatTensor(mask).permute(2, 0, 1)[None, :, :, :].cuda()

    if not include_bkg:
  
        out = out * mask + 0 * (1.0 - mask)
        gt = gt * mask + 0 * (1.0 - mask)

    out_save = out[0].permute(1, 2, 0).cpu().numpy() * 255.
    gt_save = gt[0].permute(1, 2, 0).cpu().numpy() * 255.
    out_save = out_save.astype('uint8')[:, :, ::-1]
    gt_save = gt_save.astype('uint8')[:, :, ::-1]

    cv2.imwrite(osp.join(images_folder, f"{frame_idx}_out.png"), out_save)
    cv2.imwrite(osp.join(images_folder, f"{frame_idx}_gt.png"), gt_save)

    # Compute metrics.
    results['psnr'].append(psnr(out, gt))
    results['ssim'].append(ssim(out, gt))
    results['lpips'].append(lpips(out * 2 - 1, gt * 2 - 1))  # Normalize to [-1,1]

print('output path: ' + output_path)
print('subject_id: ' + subject_id)
print('include_bkg: ' + str(include_bkg))
print({k: torch.FloatTensor(v).mean() for k, v in results.items()})
