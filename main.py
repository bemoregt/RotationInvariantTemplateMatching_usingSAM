import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import torch

MASK_COLOR = (255, 0, 0)

# Load the Lena image
image_path = "/Users/user1/Downloads/PL테크 Sensing Tab 촬상 이미지(20231215)/Vision Align Ⅱ/1_2.bmp"

image = cv2.imread(image_path)
# 새로운 크기 설정
width = 512
height = 512
dim = (width, height)

# 이미지 크기 조정
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)


original_image = image.copy()  # Keep a copy of the original image

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "/Users/user1/Downloads/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(image)

template = None  # Initialize the template

def make_mask_2_img(mask):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * np.array(MASK_COLOR).reshape(1, 1, -1)
    mask_image = mask_image.astype(np.uint8)
    return mask_image

def on_mouse_move(event):
    global image, template

    input_point = np.array([[event.x, event.y]])
    input_label = np.array([1])

    mask, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    mask_img = make_mask_2_img(mask)
    gray_mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        template = image[y:y+h, x:x+w]  # Create the template from the original image within the bounding rectangle

    image_rgb = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)

    im = Image.fromarray(image_rgb)
    img = ImageTk.PhotoImage(im)

    img_label_proc.img = img
    img_label_proc.config(image=img)

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image

def on_key_press(event):
    global image, template, original_image

    if event.char == 'm' and template is not None:
        # 매칭된 위치를 저장할 리스트
        locations = []

        # 원본 이미지에 템플릿 매칭을 위한 회전된 템플릿 생성
        for angle in range(0, 180, 1):  # 0도에서 360도까지 10도씩 회전
            # 템플릿 회전
            (h, w) = template.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_template = cv2.warpAffine(template, M, (w, h))

            # 회전된 템플릿에 대해 템플릿 매칭 실행
            res = cv2.matchTemplate(image, rotated_template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.6  # 매칭 임계값 설정
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):  # 임계값을 넘는 위치 찾기
                locations.append((pt[0], pt[1], pt[0] + w, pt[1] + h))

        # 겹치는 사각형을 처리하기 위해 cv2.groupRectangles 사용
        locations, _ = cv2.groupRectangles(locations, groupThreshold=1, eps=0.5)
        for (x1, y1, x2, y2) in locations:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 결과 이미지를 RGB로 변환하고 화면에 표시
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(image_rgb)
        img = ImageTk.PhotoImage(im)
        img_label_proc.img = img
        img_label_proc.config(image=img)

    if event.char == 'r':  # Reload the original image
        image = original_image.copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        im = Image.fromarray(image_rgb)
        img = ImageTk.PhotoImage(im)

        img_label_orig.img = img
        img_label_orig.config(image=img)

        img_label_proc.img = img
        img_label_proc.config(image=img)


root = tk.Tk()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
im = Image.fromarray(image_rgb)
img = ImageTk.PhotoImage(im)

img_label_orig = tk.Label(root, image=img)
img_label_orig.grid(row=0, column=0)

img_label_proc = tk.Label(root)
img_label_proc.grid(row=0, column=1)

root.bind("<Button-1>", on_mouse_move)
root.bind("<Key>", on_key_press)

root.mainloop()