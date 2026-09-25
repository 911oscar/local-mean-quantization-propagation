# Hardware reproduction

Original measurements: aarch64 host, Ascend910 logical device (64 GiB), PyTorch 2.7.1+cpu, torch_npu 2.7.1.post4, CANN 9.0.0, two CPU threads. The legacy quantized-convolution backend is device-specific; a CPU run is not a substitute for native timing. Use your own authorized isolated device and avoid contention for formal measurements.

Supply trusted assets in `L1_ASSET_DIR` (default `assets`):
- `cifar10.tar.gz`, SHA256 `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`, official CIFAR-10 Python archive, MD5 `c58f30108f718f92721af3b95e74349a`.
- `resnet20.pt`, SHA256 `4118986f0df73003d572b0e397f0ac7b3f60af1f31aff3d2da164536e36f6ec8`, from chenyaofo/pytorch-cifar-models. Architecture source revision `786c16252c0fc58ee9adac063f8337cc4a7a497a`.
- `assets.json`: list of objects with `name` and `sha256` for these two files. The loader verifies hashes before reading them. Assets are not bundled.

Normalize RGB/255 by means (.4914,.4822,.4465) and standard deviations (.2023,.1994,.2010), without test augmentation. `L1_OUTPUT_DIR` sets the writable device-cache root.

Run from this repository, each with a new output directory:
```
python3 code/public_mechanism.py --output runs/mechanism
python3 code/public_task.py --output runs/task
L1_LEARNED_PARAMETERS=runs/task/learned_parameters.npz python3 code/public_cost.py --output runs/cost
```
These runs produce new outputs, not the packaged records. Commands may be expensive; they were not rerun for the minor revision. The published copies preserve the validated implementation with configurable local paths; the obsolete timing loop was removed from the task script. Exact backend behavior may differ on other hardware/software. Code is not an optimized deployment algorithm. For task preparation, ridge is 1e-3 × trace(C_fit)/9 + 1e-12 per channel; complete FP32 teachers are extra offline information.
