# 🔍 PyTorch Attention Modules

A collection of modern attention mechanisms for CNN and Vision Transformers, implemented in PyTorch.

This repository provides plug-and-play attention modules that can be easily integrated into deep learning models.

---

## 🚀 Supported Attention Mechanisms

| Module | Paper | Year | Purpose |
|--------|--------|------|--------|
| SE | Squeeze-and-Excitation Networks | 2018 | Channel attention |
| CBAM | Convolutional Block Attention Module | 2018 | Channel + Spatial attention |
| ECA | Efficient Channel Attention | 2020 | Lightweight channel attention |
| BAM | Bottleneck Attention Module | 2018 | Spatial + channel refinement |
| SKNet | Selective Kernel Networks | 2019 | Adaptive receptive field |
| GCNet | Global Context Network | 2019 | Global context modeling |
| SIMAM | Simple Attention Module | 2021 | Parameter-free attention |
| GAM | Global Attention Mechanism | 2021 | Cross-dimension interaction |
| LSK | Large Selective Kernel | 2023 | Large kernel spatial modeling |
| MobileViT Attention | MobileViT | 2022 | Lightweight transformer attention |
| KAN | Kolmogorov-Arnold Network | 2024 | Functional representation learning |

---

## 📦 Installation

```bash
git clone https://github.com/zls0529/pytorch-attention-modules.git
cd pytorch-attention-modules
pip install -r requirements.txt