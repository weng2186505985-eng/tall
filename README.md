# Few-Shot Fake Video Detection System

## 简介
本项目是一个旨在解决小样本场景下深度伪造视频检测的研究系统。它结合了视觉时空不一致性分析（TALL-Swin）和音视频同步异常检测（SyncNet）。

## 开始使用

### 环境要求
- Python 3.8+
- PyTorch / torchvision
- FFmpeg (必须安装在系统路径中)
- OpenCV
- tqdm

### 1. 数据预处理
使用 `scripts/preprocess_ffplusplus.py` 来处理 FaceForensics++ 数据集并生成小样本任务集。

```bash
python scripts/preprocess_ffplusplus.py
```

### 2. 核心架构
- `models/tall_swin.py`: 视觉流 backbone。
- `models/sync_net.py`: 音频流 backbone。
- `scripts/preprocess_ffplusplus.py`: 数据清洗与元学习任务构建。

## 开发路线图
详见 [task.md](artifact/task.md)
