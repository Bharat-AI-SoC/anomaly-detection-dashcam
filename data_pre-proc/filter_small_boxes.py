import os

MIN_AREA = 0.01  # 1% of image area
BASE_DIR = "RDD2022_WORKING/RDD_SPLIT"
SPLITS = ["train", "val", "test"]

for split in SPLITS:
    label_dir = os.path.join(BASE_DIR, split, "labels")

    for file in os.listdir(label_dir):
        path = os.path.join(label_dir, file)
        kept = []

        with open(path, "r") as f:
            for line in f:
                cls, x, y, w, h = map(float, line.split())
                if w * h >= MIN_AREA:
                    kept.append(line)

        with open(path, "w") as f:
            f.writelines(kept)
