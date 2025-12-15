
# InSorg: An Instance Segmentation Method Tailored for Irregular Shapes and Small-Scale Organoids


## 📝Introduction

InSorg is an intelligent organoid image analysis platform integrated with state-of-the-art computer vision and deep learning techniques. It enables high-throughput segmentation, detection, and quantitative analysis of brightfield organoid images, automatically extracts key biological metrics, supports multi-modal image inputs, and thus significantly improves the efficiency of workflows in cancer research and personalized medicine.

<div align="center">
  <img 
    src="https://github.com/yyyhttp/InSorg/blob/main/Picture/InSorg.png" 
    alt="InSorg Image" 
    style="max-width: 100%; height: auto; max-height: 500px;"  <!-- 核心样式 -->

</div>

## 🧪Visualizations

<div align="center">
  <img 
    src="https://github.com/yyyhttp/InSorg/blob/main/Picture/data2stomach.png" 
    alt="InSorg 项目 logo" 
    style="max-width: 100%; height: auto; max-height: 500px;"  <!-- 核心样式 -->
  
</div>

## 📦Getting Started

<img src="https://img.shields.io/badge/python-3.8.x%20|%203.11.x-blueviolet" alt="Python 3.8.x | 3.10.x">

---

### • Environment Setup

```bash

# Environment and Dependency Installation Workflow (InSorg Project)
# 1. Create a conda virtual environment named InSorg_env with Python 3.8 (core runtime environment for the project)
conda create -n InSorg_env python=3.8
# 2. Activate the InSorg_env environment (all subsequent installations are executed in this isolated environment to avoid global dependency conflicts)
conda activate InSorg_env
# 3. Install the PyTorch framework (core dependency for deep learning computations; the latest compatible version for Python 3.8 will be installed if no specific version is specified)
pip3 install torch 
# 4. Install and upgrade the OpenMIM tool (package management tool for OpenMMLab projects, facilitating the installation of MM-series libraries)
pip install -U openmim
# 5. Install MMEngine via mim (MMEngine is the foundational runtime framework of OpenMMLab, providing universal engineering capabilities)
mim install mmengine 
# 6. Install MMCV (fundamental computer vision library of OpenMMLab, offering core CV operators and tools)
pip install mmcv
# 7. Batch install remaining project dependencies (requirements.txt is the custom dependency list of the project, including all packages not installed individually)
pip install -r requirements.txt
```



## ➡️Usage

---
### • Dataset Preparation
Requests for access to the full dataset should be directed to the corresponding author.
- The dataset structure is organized as follows:
```bash
InSorg/
├── data/
│   ├── coco/                      # Root directory for COCO-format dataset
│   │   ├── train/                 # Training set images (aligned with original train set, following COCO naming conventions)
│   │   │   ├── train.image        # Training set image list file
│   │   ├── val/                   # Validation set images (aligned with original val set)
│   │   │   ├── val.image          # Validation set image list file
│   │   ├── test/                  # Test set images (aligned with original test set)
│   │   │   ├── test.iamge         # Test set image list file
│   │   ├── annotations/           # COCO-format annotation files (core: JSON files containing instance segmentation masks)
│   │   │   ├── instances_train2017.json  # Instance segmentation annotations for training set (replaces original label/train/)
│   │   │   ├── instances_val2017.json    # Instance segmentation annotations for validation set (replaces original label/val/)
│   │   │   └── instances_test2017.json   # Instance segmentation annotations for test set (optional, replaces original label/test/)
```

- Execute the following command to generate data set splits:

```bash
# This switches to the OrgTrans environment for package installation
conda activate orgtrans_env

# Navigate to the datasets directory
cd datasets

# You can modify the instructions to choose any proportion of fully-supervised labels, such as 1%, 5%, or any other value. Here’s how you can update it:
# Make sure that 'data' and 'create_txt.py' are in the same directory.
python create_txt.py
```



### • Training

```shell script
# Navigate to the supervised training configuration directory
cd OrgTrans/configs/sup
# Run supervised training using the 'sup.yaml' configuration file
python train.py --cfg configs/sup/sup.yaml

# Navigate to the semi-supervised training configuration directory
cd OrgTrans/configs/ssod
# Run semi-supervised training using the 'transfer_ssod.yaml' configuration file and load the weights from supervised training
python train.py --cfg configs/ssod/transfer_ssod.yaml
```

### • Testing the Semi-Supervised Model

```shell script
python val.py --cfg configs/ssod/transfer_ssod.yaml --weights 
```



## ⚖️License

OrgLine is released under the [MIT License](MIT-License.txt), a permissive open-source license that allows for free use, modification, distribution, and private use of the software. This license requires that the original copyright notice and permission notice be included in all copies or substantial portions of the software.



## 👍Acknowledgement

We would like to thank the authors of [efficientteacher](https://github.com/AlibabaResearch/efficientteacher).



## 📃Citation

If you use this codebase in your research or project, please cite:

