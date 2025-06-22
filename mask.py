# 必要なパッケージ: torch, Pillow, numpy, opencv-python, segment_anything
# 画像パスとモデルパスを定数で指定
IMAGE_PATH = "assets/frame_001.png"  # ここを任意の画像パスに変更
MODEL_PATH = "models/sam_vit_h_4b8939.pth"  # ViT-Hの例
MODEL_TYPE = "vit_h"  # vit_h, vit_l, vit_b から選択

import torch
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import cv2
from segment_anything import sam_model_registry, SamPredictor

# 画像読み込み
image = np.array(Image.open(IMAGE_PATH).convert("RGB"))

# SAMモデルのロード
sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_PATH)
predictor = SamPredictor(sam)
predictor.set_image(image)

# GUIセットアップ
def on_click(event):
    x, y = event.x, event.y
    input_point = np.array([[x, y]])
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )
    mask = masks[0]
    # マスクを画像に重ねる
    mask_img = image.copy()
    mask_color = np.zeros_like(mask_img)
    mask_color[mask] = [0, 255, 0]  # 緑色
    overlay = cv2.addWeighted(mask_img, 0.7, mask_color, 0.3, 0)
    show_image(overlay)

def show_image(img_np):
    img_pil = Image.fromarray(img_np)
    img_tk = ImageTk.PhotoImage(img_pil)
    panel.config(image=img_tk)
    panel.image = img_tk

root = tk.Tk()
root.title("SAM Mask Demo")

img_pil = Image.fromarray(image)
img_tk = ImageTk.PhotoImage(img_pil)
panel = tk.Label(root, image=img_tk)
panel.pack()
panel.bind("<Button-1>", on_click)

root.mainloop()
