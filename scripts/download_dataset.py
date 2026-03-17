import kagglehub
import os
import shutil
import time
from pathlib import Path

# ==========================================
# Phase: Dataset Acquisition (Proxy & Custom Cache)
# ==========================================

# 设置缓存目录 (D盘)
os.environ["KAGGLEHUB_CACHE"] = "d:/bishe/kagglehub_cache"
NEW_CACHE_DIR = Path(os.environ["KAGGLEHUB_CACHE"])
NEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 设置目标目录
TARGET_DIR = Path("d:/bishe/Yuhonghai/deepfake_detection/data/raw")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# 配置代理 (7897)
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

print(f"Starting download of FaceForensics++ c23 dataset (Proxy: 7897, Cache: {NEW_CACHE_DIR})...")

max_retries = 50
retry_delay = 10
downloaded_path = None

# 下载循环：处理超时并支持自动恢复
for attempt in range(max_retries):
    try:
        downloaded_path = kagglehub.dataset_download(
            "xdxd003/ff-c23"
        )
        print("\nDownload Successful!")
        break
    except Exception as e:
        print(f"\n[尝试 {attempt + 1}/{max_retries}] 下载中断: {e}")
        if attempt < max_retries - 1:
            print(f"{retry_delay} 秒后尝试自动恢复下载...")
            time.sleep(retry_delay)
        else:
            print("已达到最大重试次数，下载失败。")
            exit(1)

# 移动文件到项目目录
if downloaded_path:
    try:
        print(f"Moving files from {downloaded_path} to {TARGET_DIR}...")
        src_path = Path(downloaded_path)
        
        for item in src_path.iterdir():
            dest_item = TARGET_DIR / item.name
            if dest_item.exists():
                if dest_item.is_dir():
                    shutil.rmtree(dest_item)
                else:
                    dest_item.unlink()
            shutil.move(str(item), str(dest_item))
            
        print(f"Dataset successfully moved to {TARGET_DIR}")
        
        # 记录路径
        with open("dataset_path.txt", "w") as f:
            f.write(str(TARGET_DIR))
            
    except Exception as e:
        print(f"\nError during moving files: {e}")
