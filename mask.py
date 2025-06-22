# 必要なパッケージ: torch, Pillow, numpy, opencv-python, segment_anything, tkinter
# 動画パスとモデルパスを定数で指定
VIDEO_PATH = "assets/videos/ComfyUI_00661_.mp4"  # ここを任意の動画パスに変更
MODEL_PATH = "models/sam_vit_h_4b8939.pth"  # ViT-Hの例
MODEL_TYPE = "vit_h"  # vit_h, vit_l, vit_b から選択

import os
import torch
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import cv2
from segment_anything import sam_model_registry, SamPredictor
import threading
from tkinter import ttk
from tqdm import tqdm

# --- 動画名・フレーム保存先決定 ---
video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
frames_dir = f"assets/{video_name}/rgb_frames"
masks_dir = f"assets/{video_name}/masks"
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

# --- 動画をフレームに分割して保存 ---
print(f"動画をフレームに分割中: {VIDEO_PATH} -> {frames_dir}")
cap = cv2.VideoCapture(VIDEO_PATH)
frame_paths = []
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
    cv2.imwrite(frame_path, frame)
    frame_paths.append(frame_path)
    frame_idx += 1
cap.release()

# --- 最初のフレームを読み込み ---
if not frame_paths:
    raise RuntimeError("動画からフレームが抽出できませんでした。動画ファイルやパスを確認してください。")
IMAGE_PATH = frame_paths[0]
image = np.array(Image.open(IMAGE_PATH).convert("RGB"))

# SAMモデルのロード
print(f"SAMモデルをロード中: {MODEL_PATH} ({MODEL_TYPE})")
sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_PATH)
predictor = SamPredictor(sam)
predictor.set_image(image)
print("SAMモデルのロード完了")

def save_mask(mask, out_path):
    # 0/1のuint8画像として保存
    mask_img = (mask.astype(np.uint8)) * 255
    Image.fromarray(mask_img).save(out_path)

def mask_centroid(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.mean()), int(ys.mean())

def process_rest_frames_centroid():
    print("[重心で処理] 開始", flush=True)
    method = "centroid"
    out_dir = os.path.join(masks_dir, method)
    os.makedirs(out_dir, exist_ok=True)
    total = len(frame_paths)
    # 最初のマスクを保存
    save_mask(mask, os.path.join(out_dir, os.path.basename(frame_paths[0])))
    prev_mask = mask.copy()
    for i, frame_path in enumerate(frame_paths[1:], 1):
        img = np.array(Image.open(frame_path).convert("RGB"))
        predictor.set_image(img)
        centroid = mask_centroid(prev_mask)
        if centroid is None:
            print(f"frame {i}: centroid not found, skipping", flush=True)
            continue
        input_point = np.array([centroid])
        input_label = np.array([1])
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        prev_mask = masks[0]
        save_mask(prev_mask, os.path.join(out_dir, os.path.basename(frame_path)))
    print(f"[重心で処理] 完了: {out_dir}", flush=True)

def iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0

def process_rest_frames_similarity():
    print("[類似度で処理] 開始", flush=True)
    method = "similarity"
    out_dir = os.path.join(masks_dir, method)
    os.makedirs(out_dir, exist_ok=True)
    # 1フレーム目のマスクを全フレームにコピーして保存（超簡素化）
    for frame_path in frame_paths:
        save_mask(mask, os.path.join(out_dir, os.path.basename(frame_path)))
        print(f"saved: {os.path.basename(frame_path)}", flush=True)
    print(f"[類似度で処理] 完了: {out_dir}", flush=True)

# --- GUIセットアップ ---
click_points = []  # [(x, y), ...]
mask = None

def run_in_thread(func):
    t = threading.Thread(target=func)
    t.start()

def on_click(event):
    # 左クリックで座標追加
    if event.num == 1:
        click_points.append((event.x, event.y))
    # 右クリックで座標削除（近い点を消す）
    elif event.num == 3 and click_points:
        dists = [np.hypot(event.x-x, event.y-y) for x, y in click_points]
        idx = np.argmin(dists)
        if dists[idx] < 20:
            click_points.pop(idx)
    update_mask()

def clear_points():
    click_points.clear()
    update_mask()

def update_mask():
    global mask
    if click_points:
        input_point = np.array(click_points)
        input_label = np.ones(len(click_points), dtype=int)
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        mask = masks[0]
        mask_img = image.copy()
        mask_color = np.zeros_like(mask_img)
        mask_color[mask] = [0, 255, 0]
        overlay = cv2.addWeighted(mask_img, 0.7, mask_color, 0.3, 0)
        for x, y in click_points:
            cv2.circle(overlay, (x, y), 5, (255, 0, 0), -1)
        show_image(overlay)
    else:
        show_image(image)

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
panel.bind("<Button-3>", on_click)

# ボタン用フレームを作成し、画像の下に横並びで配置
frame_buttons = tk.Frame(root)
frame_buttons.pack()

btn_clear = tk.Button(frame_buttons, text="選択をリセット", command=clear_points)
btn_clear.pack(side=tk.LEFT)
btn_centroid = tk.Button(frame_buttons, text="重心で処理", command=lambda: run_in_thread(process_rest_frames_centroid))
btn_centroid.pack(side=tk.LEFT)
btn_sim = tk.Button(frame_buttons, text="類似度で処理", command=lambda: run_in_thread(process_rest_frames_similarity))
btn_sim.pack(side=tk.LEFT)

root.mainloop()

# --- ここから後続処理の実装 ---
def run_in_thread(func):
    t = threading.Thread(target=func)
    t.start()

def save_mask(mask, out_path):
    # 0/1のuint8画像として保存
    mask_img = (mask.astype(np.uint8)) * 255
    Image.fromarray(mask_img).save(out_path)

def mask_centroid(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.mean()), int(ys.mean())

def process_rest_frames_centroid():
    print("[重心で処理] 開始", flush=True)
    method = "centroid"
    out_dir = os.path.join(masks_dir, method)
    os.makedirs(out_dir, exist_ok=True)
    total = len(frame_paths)
    # 最初のマスクを保存
    save_mask(mask, os.path.join(out_dir, os.path.basename(frame_paths[0])))
    prev_mask = mask.copy()
    for i, frame_path in enumerate(tqdm(frame_paths[1:], desc="centroid", unit="frame"), 1):
        img = np.array(Image.open(frame_path).convert("RGB"))
        predictor.set_image(img)
        centroid = mask_centroid(prev_mask)
        if centroid is None:
            print(f"frame {i}: centroid not found, skipping", flush=True)
            continue
        input_point = np.array([centroid])
        input_label = np.array([1])
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        prev_mask = masks[0]
        save_mask(prev_mask, os.path.join(out_dir, os.path.basename(frame_path)))
    print(f"[重心で処理] 完了: {out_dir}", flush=True)

def iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0

def process_rest_frames_similarity():
    print("[類似度で処理] 開始", flush=True)
    method = "similarity"
    out_dir = os.path.join(masks_dir, method)
    os.makedirs(out_dir, exist_ok=True)
    # 1フレーム目のマスクを全フレームにコピーして保存（超簡素化）
    for frame_path in frame_paths:
        save_mask(mask, os.path.join(out_dir, os.path.basename(frame_path)))
        print(f"saved: {os.path.basename(frame_path)}", flush=True)
    print(f"[類似度で処理] 完了: {out_dir}", flush=True)
