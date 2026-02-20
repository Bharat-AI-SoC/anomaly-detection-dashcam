import os
import cv2
import numpy as np
import random

BASE_DIR = "RDD2022_WORKING/RDD_SPLIT"
SPLITS = ["train"]   # augment ONLY train
OUT_SUFFIX = "_aug"

def random_brightness_contrast(img):
    alpha = random.uniform(0.8, 1.2)   # contrast
    beta = random.randint(-20, 20)      # brightness
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def motion_blur(img):
    k = random.choice([3, 5])
    kernel = np.zeros((k, k))
    kernel[int((k-1)/2), :] = np.ones(k)
    kernel /= k
    return cv2.filter2D(img, -1, kernel)

for split in SPLITS:
    img_dir = os.path.join(BASE_DIR, split, "images")
    label_dir = os.path.join(BASE_DIR, split, "labels")

    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))

        img = cv2.imread(img_path)
        if img is None:
            continue

        aug = img.copy()

        if random.random() < 0.5:
            aug = random_brightness_contrast(aug)

        if random.random() < 0.3:
            aug = motion_blur(aug)

        new_img_name = img_name.replace(".jpg", f"{OUT_SUFFIX}.jpg")
        new_label_name = img_name.replace(".jpg", f"{OUT_SUFFIX}.txt")

        cv2.imwrite(os.path.join(img_dir, new_img_name), aug)
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
            with open(os.path.join(label_dir, new_label_name), "w") as f:
                f.writelines(lines)
