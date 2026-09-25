# Local mean correction: public-model companion

Version 1.0.0. Public companion materials; no license is supplied. See [notice](NOTICE.md). No paper manuscript or preprint is included.

This standalone package supports a study of neural-network quantization-error propagation on **ResNet-20 / CIFAR-10**. Local fidelity, same-input downstream fidelity, full-FP32 fidelity, label cross-entropy and accuracy are distinct outcomes. There are 120 mechanism images × 6 interfaces (720 dependent interventions), not 720 independent images. Task metrics reuse 1000 fixed test images.

## Read and verify
- [Protocol and provenance](docs/PROTOCOL.md)
- [Results and numerical definitions](docs/RESULTS.md)
- [Hardware reproduction](docs/REPRODUCE.md)
- `data/`: scalar records and summaries; `code/`: public-model implementation and lightweight verification.
- Run `python3 code/verify_records.py` without torch or a device to verify file hashes and counts. This verifies supplied records, not an independent rerun of inference.

## Availability and exclusions
CIFAR-10 is obtained separately from its [provider](https://www.cs.toronto.edu/~kriz/cifar.html). The [trained model](https://github.com/chenyaofo/pytorch-cifar-models) is also obtained separately. No images or weights are redistributed here. Meteorological data used in the study were provided by the project funded under Guangdong S&T Program 2025B0101080001; meteorological data and derived records are not part of this release. No request-based access promise is made.

The repository contains a fresh, independently curated file set. It excludes the internal repository/history, private reviews, meteorological assets, patent materials, server configuration, credentials, fonts and the paper itself. These exclusions do not change which experiments support the paper.

中文说明：本包仅包含 CIFAR-10 公开任务的标量证据、方法代码和复算说明；逐层干预不是独立样本。气象数据由项目 2025B0101080001 提供，本包不公开气象数据或派生资产，也不承诺申请即可取得。代码运行需要相应硬件，现有记录检查不等同重新复现实验。发布不构成论文投稿或预印本发布。
