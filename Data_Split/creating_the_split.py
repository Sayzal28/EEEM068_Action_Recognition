import shutil
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Paths
original_root = "/content/drive/MyDrive/AML/HMDB_simp"
subset_root = "/content/drive/MyDrive/AML/dataset/test"
csv_path = "/content/drive/MyDrive/AML/split/test.csv"

def copy_video(row):
    """Copy a single video"""
    label, video = row['class'], row['video_name']

    src_path = os.path.join(original_root, label, video)
    dst_path = os.path.join(subset_root, label, video)

    if os.path.exists(src_path):
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        return True
    else:
        print(f"Warning: {src_path} does not exist.")
        return False

# Load CSV and copy files
df = pd.read_csv(csv_path)

def parallel_copy(max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(copy_video, [row for _, row in df.iterrows()]))

    success_count = sum(results)
    print(f"Copied {success_count}/{len(df)} videos")

# Run parallel copy
if __name__ == "__main__":
    print(f"Copying {len(df)} videos...")
    parallel_copy()
    print("Done!")
