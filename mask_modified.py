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
import scipy.ndimage

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

# 重心処理モードは削除

def iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0

def mask_sample_points(mask, num_points=8):
    """マスクのすべての輪郭から等間隔でnum_points個ずつ点をサンプリング"""
    import cv2
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    points = []
    for cnt in contours:
        if len(cnt) < num_points:
            pts = cnt[:, 0, :]
        else:
            idxs = np.linspace(0, len(cnt)-1, num_points, dtype=int)
            pts = cnt[idxs, 0, :]
        points.extend([tuple(pt) for pt in pts])
    return points

def get_connected_components(mask):
    labeled, n = scipy.ndimage.label(mask)
    masks = [(labeled == i+1) for i in range(n)]
    return masks

def process_rest_frames_similarity():
    print("[類似度で処理:領域ごと追跡] 開始", flush=True)
    method = "similarity"
    out_dir = os.path.join(masks_dir, method)
    main_dir = os.path.join(out_dir, "main")
    other_dir = os.path.join(out_dir, "other")
    main_only_dir = os.path.join(out_dir, "main_only")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(main_dir, exist_ok=True)
    os.makedirs(other_dir, exist_ok=True)
    os.makedirs(main_only_dir, exist_ok=True)
    # 1フレーム目のマスクを保存
    save_mask(mask_main, os.path.join(main_dir, os.path.basename(frame_paths[0])))
    save_mask(mask_other, os.path.join(other_dir, os.path.basename(frame_paths[0])))
    save_mask(mask_main_only, os.path.join(main_only_dir, os.path.basename(frame_paths[0])))
    save_mask(mask_main_only, os.path.join(out_dir, os.path.basename(frame_paths[0])))
    prev_mask = mask_main_only.copy()
    for i, frame_path in enumerate(frame_paths[1:], 1):
        img = np.array(Image.open(frame_path).convert("RGB"))
        predictor.set_image(img)
        prev_regions = get_connected_components(prev_mask)
        curr_masks = []
        for region in prev_regions:
            centroid = mask_centroid(region)
            if centroid is None:
                continue
            input_point = np.array([centroid])
            input_label = np.array([1])
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
            # regionとIoU最大のマスクを選ぶ
            ious = [iou(region, m) for m in masks]
            best_idx = int(np.argmax(ious))
            best_mask = masks[best_idx]
            curr_masks.append(best_mask)
        if curr_masks:
            combined_mask = np.any(curr_masks, axis=0)
        else:
            combined_mask = np.zeros_like(prev_mask)
        # main: 最初の領域に最もIoUが高いものをmainとする
        if curr_masks:
            prev_main = get_connected_components(prev_mask)[0]
            ious_main = [iou(prev_main, m) for m in curr_masks]
            main_idx = int(np.argmax(ious_main))
            main_mask = curr_masks[main_idx]
            # other: main以外
            other_masks = [m for j, m in enumerate(curr_masks) if j != main_idx]
            if other_masks:
                other_mask = np.any(other_masks, axis=0)
            else:
                other_mask = np.zeros_like(main_mask)
            # main_only: main - other
            main_only_mask = np.logical_and(main_mask, np.logical_not(other_mask))
        else:
            main_mask = np.zeros_like(prev_mask)
            other_mask = np.zeros_like(prev_mask)
            main_only_mask = np.zeros_like(prev_mask)
        prev_mask = combined_mask
        # 保存
        save_mask(main_mask, os.path.join(main_dir, os.path.basename(frame_paths[i])))
        save_mask(other_mask, os.path.join(other_dir, os.path.basename(frame_paths[i])))
        save_mask(main_only_mask, os.path.join(main_only_dir, os.path.basename(frame_paths[i])))
        save_mask(main_only_mask, os.path.join(out_dir, os.path.basename(frame_paths[i])))
        print(f"frame {i}: similarity mask saved (regions={len(curr_masks)})", flush=True)
    print(f"[類似度で処理] 完了: {out_dir}", flush=True)

# --- GUIセットアップ ---
root = tk.Tk()
root.title("SAM Mask Demo")

# --- GUIセットアップ ---
click_points_main = []  # 追跡対象用 [(x, y), ...]
click_points_other = [] # 除外用 [(x, y), ...]
mask_main = None
mask_other = None
mask_main_only = None
click_mode = tk.StringVar(value="main")  # ここで初期化

# マスクの色設定（RGB形式）
main_mask_color = [0, 255, 0]  # 緑色（デフォルト）
other_mask_color = [255, 0, 0]  # 赤色（デフォルト）
main_only_mask_color = [0, 255, 255]  # シアン色（デフォルト）

def clear_points():
    click_points_main.clear()
    click_points_other.clear()
    update_mask()

def on_click(event):
    # 左クリックで座標追加
    if event.num == 1:
        if click_mode.get() == "main":
            click_points_main.append((event.x, event.y))
        else:
            click_points_other.append((event.x, event.y))
    # 右クリックで座標削除（近い点を消す）
    elif event.num == 3:
        if click_mode.get() == "main" and click_points_main:
            dists = [np.hypot(event.x-x, event.y-y) for x, y in click_points_main]
            idx = np.argmin(dists)
            if dists[idx] < 20:
                click_points_main.pop(idx)
        elif click_mode.get() == "other" and click_points_other:
            dists = [np.hypot(event.x-x, event.y-y) for x, y in click_points_other]
            idx = np.argmin(dists)
            if dists[idx] < 20:
                click_points_other.pop(idx)
    update_mask()

def update_mask():
    global mask_main, mask_other, mask_main_only
    # main
    if click_points_main:
        input_point = np.array(click_points_main)
        input_label = np.ones(len(click_points_main), dtype=int)
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        mask_main = masks[0]
        print(f"Main mask pixels: {mask_main.sum()}")
    else:
        mask_main = np.zeros(image.shape[:2], dtype=bool)
    # other
    if click_points_other:
        input_point = np.array(click_points_other)
        input_label = np.ones(len(click_points_other), dtype=int)
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        mask_other = masks[0]
        print(f"Other mask pixels: {mask_other.sum()}")
    else:
        mask_other = np.zeros(image.shape[:2], dtype=bool)
    # main_only
    mask_main_only = np.logical_and(mask_main, np.logical_not(mask_other))
    print(f"Main-only mask pixels: {mask_main_only.sum()}")
    
    # マスクを画像に重ねて表示
    overlay = image.copy()
    # main マスク
    mask_color = np.zeros_like(overlay)
    mask_color[mask_main] = main_mask_color  # 設定した色
    overlay = cv2.addWeighted(overlay, 0.7, mask_color, 0.3, 0)
    
    # other マスク
    if np.any(mask_other):
        mask_color2 = np.zeros_like(overlay)
        mask_color2[mask_other] = other_mask_color  # 設定した色
        overlay = cv2.addWeighted(overlay, 0.7, mask_color2, 0.3, 0)
    
    # main_only マスク
    mask_color3 = np.zeros_like(overlay)
    mask_color3[mask_main_only] = main_only_mask_color  # 設定した色
    overlay = cv2.addWeighted(overlay, 0.7, mask_color3, 0.3, 0)
    
    # クリックポイントを表示
    for x, y in click_points_main:
        cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)  # 青色の点
    for x, y in click_points_other:
        cv2.circle(overlay, (x, y), 5, (255, 255, 0), -1)  # 黄色の点
    
    # 更新された画像を表示
    img_pil = Image.fromarray(overlay)
    img_tk = ImageTk.PhotoImage(img_pil)
    panel.config(image=img_tk)
    panel.image = img_tk
    
    # デバッグ情報を表示
    print(f"マスク更新: main={np.sum(mask_main)} pixels, other={np.sum(mask_other)} pixels, main_only={np.sum(mask_main_only)} pixels")

# 色設定用の関数
def choose_color(color_var, title):
    from tkinter import colorchooser
    color = colorchooser.askcolor(title=title)[0]
    if color:
        color_var[0] = color[0]
        color_var[1] = color[1]
        color_var[2] = color[2]
        update_mask()

# ボタン用フレーム
frame_buttons = tk.Frame(root)
frame_buttons.pack()
btn_clear = tk.Button(frame_buttons, text="選択をリセット", command=clear_points)
btn_clear.pack(side=tk.LEFT)
btn_color_main = tk.Button(frame_buttons, text="追跡色設定", command=lambda: choose_color(main_mask_color, "追跡対象の色を選択"))
btn_color_main.pack(side=tk.LEFT)
btn_color_other = tk.Button(frame_buttons, text="除外色設定", command=lambda: choose_color(other_mask_color, "除外対象の色を選択"))
btn_color_other.pack(side=tk.LEFT)
btn_color_main_only = tk.Button(frame_buttons, text="最終マスク色設定", command=lambda: choose_color(main_only_mask_color, "最終マスクの色を選択"))
btn_color_main_only.pack(side=tk.LEFT)
btn_sim = tk.Button(frame_buttons, text="類似度で処理", command=lambda: run_in_thread(process_rest_frames_similarity))
btn_sim.pack(side=tk.LEFT)

# 画像表示用パネル
panel = tk.Label(root)
panel.pack()
panel.bind("<Button-1>", on_click)
panel.bind("<Button-3>", on_click)

# 初期画像を表示
img_pil = Image.fromarray(image)
img_tk = ImageTk.PhotoImage(img_pil)
panel.config(image=img_tk)
panel.image = img_tk

# --- スレッド実行用関数 ---
def run_in_thread(func):
    t = threading.Thread(target=func)
    t.start()

root.mainloop()
