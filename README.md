# Facial Expression Recognition with ResNet + Attention

本项目是在原 facial-expression-recognition 代码基础上重构的 PyTorch 表情识别课程大作业版本。原始 `model_CNN.py`、`model_VGG.py`、`model_ResNet.py` 已保留，可继续复现实验；新增代码位于 `src/`、`configs/`、`scripts/`、`app/`，用于更规范地训练、评估、可视化和展示 ResNet + 注意力机制实验。

## 项目背景

面部表情识别需要从人脸图像中区分 angry、disgust、fear、happy、sad、surprise、neutral 七类情绪。FER2013 图像分辨率低、类别不均衡明显，适合展示机器视觉中的数据增强、迁移学习、注意力机制、模型评估与可解释性分析。

## 原项目问题

- 训练脚本分散在 `model_CNN.py`、`model_VGG.py`、`model_ResNet.py`，参数和数据路径写死。
- 旧脚本在 import 时会直接加载数据，不便于统一实验管理。
- 缺少标准化的 `metrics.csv`、classification report、confusion matrix 和预测明细。
- 没有统一配置文件、随机种子、类别不均衡处理、Grad-CAM 和演示前端。

## 改进方案

- 用 `configs/*.yaml` 管理 CNN baseline、ResNet18、ResNet18+CBAM、ResNet18+CBAM+Focal Loss。
- 用 `src/train.py` 和 `src/evaluate.py` 统一训练与评估。
- 支持 ImageFolder 数据和 FER2013 CSV 转 ImageFolder。
- 支持 ImageNet 预训练 ResNet、`grayscale_to_rgb=True`、SE/CBAM 注意力、Focal Loss、Label Smoothing、class weight 与 WeightedRandomSampler。
- 自动导出曲线图、混淆矩阵、每类 F1 柱状图、Grad-CAM 示例和 Gradio demo。

## 环境搭建

推荐 Python 3.10+。CPU 版本可直接安装：

```bash
pip install -r requirements.txt
```

如果在 CUDA 服务器上训练，请按服务器 CUDA 版本安装 PyTorch，例如：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pillow matplotlib pyyaml gradio opencv-python
```

## 数据准备

### ImageFolder 格式

支持如下结构：

```text
face_images/
  train/
    angry/
    disgust/
    ...
  valid/
    angry/
    ...
  test/
    angry/
    ...
```

也兼容旧项目常见 split 名称，如 `train_set`、`verify_set`、`resnet_train_set`、`resnet_vaild_set`。

### FER2013 CSV 转换

FER2013 CSV 应包含 `emotion,pixels,Usage` 三列：

```bash
python -m scripts.prepare_fer2013 --csv dataset/fer2013.csv --output_dir face_images --image_size 48
```

转换后 `Training` -> `train`，`PublicTest` -> `valid`，`PrivateTest` -> `test`。

## 本地训练

CNN baseline：

```bash
python -m src.train --config configs/cnn_baseline.yaml --data_dir face_images
```

ResNet18：

```bash
python -m src.train --config configs/resnet18.yaml --data_dir face_images
```

ResNet18 + CBAM + Focal Loss：

```bash
python -m src.train --config configs/resnet18_cbam_focal.yaml --data_dir face_images --epochs 60 --batch_size 64 --lr 0.0003
```

训练产物保存在 `outputs/<run_name>/`：

- `best.pt`：按 validation macro F1 保存的最佳模型
- `last.pt`：最后一个 epoch
- `metrics.csv`：每轮 `train_loss`、`val_loss`、`train_acc`、`val_acc`、`macro_f1`、`balanced_acc`、`lr`
- `config.yaml`：本次运行配置备份

## HPC 训练

项目已提供 Slurm 脚本，默认配置为 1 张 GPU、8 CPU、32G 内存、8 小时，日志输出到 `logs/slurm-%j.out` 和 `logs/slurm-%j.err`。

默认 HPC 项目路径：

```bash
/share/home/u20526/czx/fer-resnet-attention
```

如果你的服务器路径不同，可以直接编辑脚本里的 `PROJECT_DIR`，也可以提交时用环境变量覆盖：

```bash
PROJECT_DIR=/your/project/path DATA_DIR=/your/project/path/face_images sbatch scripts/train_resnet18_cbam_focal.sbatch
```

单独提交实验：

```bash
sbatch scripts/train_cnn_baseline.sbatch
sbatch scripts/train_resnet18.sbatch
sbatch scripts/train_resnet18_cbam.sbatch
sbatch scripts/train_resnet18_cbam_focal.sbatch
```

并行提交四组训练：

```bash
bash scripts/run_all_hpc.sh
```

四组训练分别输出到：

- `outputs/cnn_baseline`
- `outputs/resnet18`
- `outputs/resnet18_cbam`
- `outputs/resnet18_cbam_focal`

训练完成后批量评估并汇总 PPT 对比图：

```bash
sbatch scripts/evaluate_all.sbatch
```

`evaluate_all.sbatch` 会依次评估已有的 `outputs/<run_name>/best.pt`，生成 `eval_metrics.json`、`classification_report.csv`、`confusion_matrix.csv`、`predictions.csv`，并调用 `scripts.collect_results` 生成：

- `assets/figures/model_comparison_table.csv`
- `assets/figures/model_comparison_macro_f1.png`
- `assets/figures/model_comparison_balanced_acc.png`
- `assets/figures/model_comparison_accuracy.png`

从 HPC 同步结果到本地：

```bash
bash scripts/sync_results_from_hpc.sh user@hpc.example.edu /share/home/u20526/czx/fer-resnet-attention .
```

## 消融实验设计

消融实验用于回答“每个改动是否真的有贡献”。本项目提供五组配置：

- `configs/ablation/resnet18_no_aug.yaml`：ResNet18，不使用训练增强。
- `configs/ablation/resnet18_aug.yaml`：加入 RandomFlip、Rotation、Affine、ColorJitter、RandomErasing。
- `configs/ablation/resnet18_cbam_aug.yaml`：在增强基础上加入 CBAM 注意力。
- `configs/ablation/resnet18_cbam_aug_focal.yaml`：在 CBAM 基础上使用 Focal Loss 和 class weight。
- `configs/ablation/resnet18_cbam_aug_focal_sampler.yaml`：再加入 WeightedRandomSampler。

Augmentation 的作用是模拟表情图像中常见的轻微姿态、光照、遮挡和局部缺失变化，降低模型对训练集细节的记忆，提升泛化能力。

CBAM 注意力的作用是从通道和空间两个维度重新加权特征，让模型更容易关注眼睛、眉毛、嘴角等表情关键区域，而不是背景或无关纹理。

Focal Loss 的作用是降低简单样本对梯度的主导，让模型把更多学习能力放在易混淆或少数类样本上，适合 FER2013 这类类别不均衡数据。

Weighted Sampler 的作用是在训练 batch 中提高少数类出现频率，缓解模型偏向 happy、neutral 等大类的问题。

不能只看 Accuracy，因为类别不均衡时模型只要偏向大类也可能得到看似不错的准确率。课程展示建议同时报告 macro F1 和 balanced accuracy：macro F1 平等衡量每个类别的 F1，balanced accuracy 平等衡量每个类别的召回率，更能反映 disgust、fear 等少数类是否被真正识别。

HPC 上并行提交五组消融训练：

```bash
bash scripts/run_ablation_hpc.sh
```

训练或评估完成后生成 PPT 对比图：

```bash
python -m scripts.plot_ablation --outputs_dir outputs --assets_dir assets/figures
```

输出包括：

- `assets/figures/ablation_table.csv`
- `assets/figures/ablation_macro_f1_bar.png`
- `assets/figures/ablation_balanced_acc_bar.png`

鲁棒性测试用于展示模型在亮度、对比度、噪声、模糊、旋转扰动下的稳定性：

```bash
python -m src.robustness_test \
  --checkpoint outputs/resnet18_cbam_focal/best.pt \
  --test_dir face_images/test \
  --output_dir outputs/resnet18_cbam_focal
```

会生成 `robustness_results.csv` 以及 `robustness_brightness.png`、`robustness_contrast.png`、`robustness_noise.png`、`robustness_blur.png`、`robustness_rotation.png`、`robustness_summary.png`。

## 评估

```bash
python -m src.evaluate --checkpoint outputs/resnet18_cbam_focal/best.pt --data_dir face_images --split test
```

评估会保存：

- `eval_metrics.json`
- `classification_report.csv`
- `confusion_matrix.csv`
- `predictions.csv`

其中 `predictions.csv` 包含 `image_path`、`label`、`pred`、`probabilities`。

## 结果可视化

```bash
python -m src.visualize_results --run_dir outputs/resnet18_cbam_focal
```

图表会同时保存到：

- `outputs/<run_name>/figures/`
- `assets/figures/`

## Grad-CAM

```bash
python -m src.gradcam_vis \
  --checkpoint outputs/resnet18_cbam_focal/best.pt \
  --images face_images/test/happy/example.png face_images/test/sad/example.png \
  --output outputs/resnet18_cbam_focal/figures/gradcam_examples.png
```

Grad-CAM 用于展示模型关注区域是否集中在眼睛、眉毛、嘴角等表情关键区域。

## PPT 可视化结果生成

生成 ResNet18 的真实测试图片预测案例图和 Grad-CAM 图：

```bash
sbatch scripts/generate_ppt_visuals.sbatch
```

脚本会读取 `outputs/resnet18/predictions.csv` 和 `face_images/test`，输出：

- `assets/figures/ppt/ppt_resnet18_correct_examples.png`
- `assets/figures/ppt/ppt_resnet18_wrong_examples.png`
- `assets/figures/ppt/ppt_resnet18_mixed_prediction_examples.png`
- `assets/figures/ppt/ppt_resnet18_gradcam_examples.png`

也可以本地单独运行：

```bash
python scripts/make_real_image_results.py \
  --predictions outputs/resnet18/predictions.csv \
  --data_dir face_images \
  --save_dir assets/figures/ppt \
  --n 12

python -m src.gradcam_vis \
  --config configs/resnet18.yaml \
  --checkpoint outputs/resnet18/best.pt \
  --data_dir face_images \
  --output_dir assets/figures/ppt \
  --num_images 16
```

如果 `predictions.csv` 仍是旧格式或不存在，请先重新评估：

```bash
python -m src.evaluate \
  --config configs/resnet18.yaml \
  --checkpoint outputs/resnet18/best.pt \
  --data_dir face_images \
  --split test \
  --output_dir outputs/resnet18
```

## 前端应用

```bash
python -m app.gradio_app --checkpoint outputs/resnet18_cbam_focal/best.pt
```

页面标题为 **Facial Expression Recognition Demo**，支持上传图片和摄像头单帧识别，输出预测表情、各类别置信度图和 Grad-CAM 可视化图。

## PPT 可用图表清单

- `train_val_loss.png`：训练/验证损失曲线
- `train_val_acc.png`：训练/验证准确率曲线
- `macro_f1_curve.png`：验证 macro F1 曲线
- `learning_rate_curve.png`：学习率曲线
- `normalized_confusion_matrix.png`：归一化混淆矩阵
- `per_class_f1_bar.png`：每类 F1 柱状图
- `model_comparison_table.csv`：不同实验对比表
- `model_comparison_macro_f1.png`：不同实验 macro F1 对比
- `model_comparison_balanced_acc.png`：不同实验 balanced accuracy 对比
- `model_comparison_accuracy.png`：不同实验 accuracy 对比
- `ablation_table.csv`：消融实验指标表
- `ablation_macro_f1_bar.png`：消融实验 macro F1 对比
- `ablation_balanced_acc_bar.png`：消融实验 balanced accuracy 对比
- `robustness_brightness.png`：亮度扰动鲁棒性曲线
- `robustness_contrast.png`：对比度扰动鲁棒性曲线
- `robustness_noise.png`：噪声扰动鲁棒性曲线
- `robustness_blur.png`：模糊扰动鲁棒性曲线
- `robustness_rotation.png`：旋转扰动鲁棒性曲线
- `robustness_summary.png`：鲁棒性汇总图
- `gradcam_examples.png`：模型可解释性示例

## 目录结构

```text
configs/                 # 实验配置
src/data/                # Dataset/DataLoader
src/models/              # CNN baseline, ResNet, SE/CBAM
src/utils/               # loss, metrics, config, seed
scripts/                 # FER2013 CSV 转换脚本
outputs/                 # 训练输出
logs/                    # HPC/训练日志
assets/figures/          # PPT 图表
app/                     # Gradio 前端
model_CNN.py             # legacy baseline, 保留
model_VGG.py             # legacy baseline, 保留
model_ResNet.py          # legacy baseline, 保留
```
