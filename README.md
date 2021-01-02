# Cifar10-High-Accuracy-Models

* Please note, all images were augmented. Without image augmentation it is so hard to make a model without overfitting.
* No augmentation leads to overfitting so easily.
* You might get better results by playing with the architecture.

| **Model Name** | Total Params | GPU vRAM Usage | Optimizer | Epochs | Each Epoch(in secs) | Train Time | Training Accuracy | Validation Accuracy | Testing Accuracy |
|-|-|-|-|-|-|-|-|-|-|
| Plain CNN v1 | 392,682 | < 4 GB | RMSProp | 289 | 35s (45ms/step) | 2 hrs 48 mins | 90.84% | 87.34% | 86.13% |
| Plain CNN v2 | 380,714 | < 4 GB | RMSProp | 332 | 25s (32ms/step) | 2 hrs 18 mins | 86.75% | 87.69% | 87.33% |
| Plain CNN v3 | 380,714 | < 4 GB | Adam | 142 | 26s (33ms/step) | 1 hrs 1 min | 81.65% | 84.46% | 84.47% |
| Plain CNN v4 | 323,146 | < 4 GB | Adam | 83 | 25s (32ms/step) | 34 mins | 88.77% | 85.67% | 84.73% |
| Plain CNN v5 | 180,778 | < 4 GB | Adam | 132 | 24s (32ms/step) | 53 mins | 85.02% | 83.13% | 83.57% |
| ResNet50 - No Weights | 45,301,762 (images upscaled) | 12.60 GB | Adam | 34 | 537s (689ms/step) | 5 hrs 4 mins | 92.95% | 90.46% | 90.1% |
| ResNet50 - Imagenet (Simple Transfer Learning) | 45,170,178 (upscaled images) | Around 9 GB | RMSProp | 74 | 148s (189ms/step) | 3 hrs 2 mins | 91.25% | 89.86% | 89.1% |
