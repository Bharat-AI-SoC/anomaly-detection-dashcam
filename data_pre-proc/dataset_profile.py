import os
import numpy as np

BASE_DIR = "RDD2022_WORKING/RDD_SPLIT"
SPLITS = ["train", "val"]

areas = []
aspect_ratios = []
images_with_boxes = 0
total_images = 0

for split in SPLITS:
    label_dir = os.path.join(BASE_DIR, split, "labels")

    for file in os.listdir(label_dir):
        total_images += 1
        path = os.path.join(label_dir, file)

        with open(path, "r") as f:
            lines = f.readlines()

        if len(lines) > 0:
            images_with_boxes += 1

        for line in lines:
            _, _, _, w, h = map(float, line.split())
            areas.append(w * h)
            aspect_ratios.append(w / h if h > 0 else 0)

print("Total images:", total_images)
print("Images with anomalies:", images_with_boxes)
print("Anomaly ratio:", round(images_with_boxes / total_images, 3))
print("Avg box area:", round(np.mean(areas), 4))
print("Median box area:", round(np.median(areas), 4))
print("Avg aspect ratio:", round(np.mean(aspect_ratios), 2))
