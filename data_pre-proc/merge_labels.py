import os

# RDD2022 class IDs to KEEP
KEEP = {3, 4}   # pothole, other corruption

BASE_DIR = "RDD2022_WORKING/RDD_SPLIT"
SPLITS = ["train", "val", "test"]

for split in SPLITS:
    label_dir = os.path.join(BASE_DIR, split, "labels")

    for file in os.listdir(label_dir):
        path = os.path.join(label_dir, file)
        new_lines = []

        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                cls = int(parts[0])

                if cls in KEEP:
                    # map to single class 0
                    new_line = "0 " + " ".join(parts[1:]) + "\n"
                    new_lines.append(new_line)

        # overwrite label file safely
        with open(path, "w") as f:
            f.writelines(new_lines)
