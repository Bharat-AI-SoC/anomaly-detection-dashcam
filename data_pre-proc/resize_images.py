import os
import cv2

BASE_DIR = "RDD2022_WORKING/RDD_SPLIT"
SPLITS = ["train", "val", "test"]

TARGET_SIZE = 416  # use 416; later you can switch to 320

for split in SPLITS:
    img_dir = os.path.join(BASE_DIR, split, "images")

    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_resized = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))
        cv2.imwrite(img_path, img_resized)
