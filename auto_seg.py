import numpy as np
from PIL import Image
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import itertools
import os

IMAGE_PATH = "path/to/your/image.png"
MODEL_PATH = "models/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"  # vit_h, vit_l, vit_b

# 画像読み込み
image = np.array(Image.open(IMAGE_PATH).convert("RGB"))

# SAMモデルのロード
sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_PATH)

# パラメータの3パターンずつ
points_per_side_list = [16, 32, 64]
pred_iou_thresh_list = [0.80, 0.88, 0.95]
stability_score_thresh_list = [0.90, 0.95, 0.99]
min_mask_region_area_list = [50, 100, 500]

param_combinations = list(itertools.product(
    points_per_side_list,
    pred_iou_thresh_list,
    stability_score_thresh_list,
    min_mask_region_area_list
))

for idx, (pps, pit, sst, mmra) in enumerate(param_combinations):
    print(f"\n=== Pattern {idx+1}/81 ===")
    print(f"points_per_side={pps}, pred_iou_thresh={pit}, stability_score_thresh={sst}, min_mask_region_area={mmra}")
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=pps,
        pred_iou_thresh=pit,
        stability_score_thresh=sst,
        min_mask_region_area=mmra
    )
    masks = mask_generator.generate(image)
    out_dir = f"sam_masks/pattern_{idx+1:02d}_pps{pps}_iou{pit}_sst{sst}_area{mmra}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"領域数: {len(masks)}")
    for i, m in enumerate(masks):
        mask_img = (m['segmentation'].astype(np.uint8)) * 255
        Image.fromarray(mask_img).save(os.path.join(out_dir, f"mask_{i:03d}.png"))