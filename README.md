# Cifar10-High-Accuracy-Models

* Please note, all images were augmented. Without image augmentation it is so hard to make a model without overfitting.
* No augmentation leads to overfitting so easily.
* You might get better results by playing with the architecture.

| **Model Name**                                 | Total Params                 | GPU vRAM Usage | Optimizer | Epochs           | Training Accuracy | Validation Accuracy | Testing Accuracy |
|------------------------------------------------|------------------------------|----------------|-----------|------------------|-------------------|---------------------|------------------|
| Plain CNN v1                                   | 392,682                      | Less than 4 GB | RMSProp   | 289              | 90.84%            | 87.34%              | 86.13%           |
| Plain CNN v2                                   | 380,714                      | Less than 4 GB | RMSProp   | 332              | 86.75%            | 87.69%              | 87.33%           |
| Plain CNN v3                                   | 380,714                      | Less than 4 GB | Adam      | 142              | 81.65%            | 84.46%              | 84.47%           |
| Plain CNN v4                                   | 323,146                      | Less than 4 GB | Adam      | 83               | 88.77%            | 85.67%              | 84.73%           |
| Plain CNN v5                                   | 180,778                      | Less than 4 GB | Adam      | 132              | 85.02%            | 83.13%              | 83.57%           |
| ResNet50 - No Weights                          | 45,301,762 (images upscaled) | 12.60 GB       | Adam      | 34 (interrupted) | 92.95%            | 90.46%              | 90.1%            |
| ResNet50 - Imagenet (Simple Transfer Learning) | 45,170,178 (upscaled images) | Around 9 GB    | RMSProp   | 74               | 91.25%            | 89.86%              | 89.1%            |
