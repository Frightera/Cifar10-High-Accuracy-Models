# Cifar10-High-Accuracy-Models

* Please note, all images were augmented. Without image augmentation it is so hard to make a model without overfitting.
* No augmentation leads to overfitting so easily.
* You might get better results by playing with the architecture.

| **Model Name**          | Plain CNN v1    | Plain CNN v2    | Plain CNN v3    | Plain CNN v4    | Plain CNN v5    | ResNet50 - No Weights        | ResNet50 - Imagenet (Simple Transfer Learning) |
|-------------------------|-----------------|-----------------|-----------------|-----------------|-----------------|------------------------------|------------------------------------------------|
| **Total Params**        | 392,682         | 380,714         | 380,714         | 323,146         | 180,778         | 45,301,762 (images upscaled) | 45,170,178 (upscaled images)                   |
| **GPU vRAM Usage**      | < 4 GB          | < 4 GB          | < 4 GB          | < 4 GB          | < 4 GB          | 12.60 GB                     | Around 9 GB                                    |
| **Optimizer**           | RMSProp         | RMSProp         | Adam            | Adam            | Adam            | Adam                         | RMSProp                                        |
| **Epochs**              | 289             | 332             | 142             | 83              | 132             | 34                           | 74                                             |
| **Each Epoch(in secs)** | 35s (45ms/step) | 25s (32ms/step) | 26s (33ms/step) | 25s (32ms/step) | 24s (32ms/step) | 537s (689ms/step)            | 148s (189ms/step)                              |
| **Train Time**          | 2 hrs 48 mins   | 2 hrs 18 mins   | 1 hrs 1 min     | 34 mins         | 53 mins         | 5 hrs 4 mins                 | 3 hrs 2 mins                                   |
| **Training Accuracy**   | 90.84%          | 86.75%          | 81.65%          | 88.77%          | 85.02%          | 92.95%                       | 91.25%                                         |
| **Validation Accuracy** | 87.34%          | 87.69%          | 84.46%          | 85.67%          | 83.13%          | 90.46%                       | 89.86%                                         |
| **Testing Accuracy**    | 86.13%          | 87.33%          | 84.47%          | 84.73%          | 83.57%          | 90.1%                        | 89.1%                                          |
