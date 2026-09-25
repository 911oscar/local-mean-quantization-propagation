# Protocol and provenance

Before the public-model outcomes, the internal protocol fixed one publicly trained ResNet-20 checkpoint, six second-convolution interfaces (residual units 0,2,3,5,6,8), the hash-selected input sets and amplitudes 0.25, 1, 4. Publication of this document is retrospective: a new public repository timestamp is **not independent proof of preregistration**. No internal Git history is distributed.

Within each class, sort training indices by SHA256 of `L1R-cal-v1:<index>`. First 24 fit completion, next 8 develop (240/80 overall); first 4 fitting images per class calibrate mechanism diagnostics. Independently sort test indices by `L1R-test-v1:<index>`: first 100 per class evaluate tasks, first 12 test mechanisms, first 4 test amplitudes. See `data/data_plan.json`. Alpha=1 overlaps the natural set. Dataset, checkpoint and normalization are fixed; no weight search or retraining is claimed.

W8A8 convolution has symmetric per-output-channel weights and dynamic per-sample activations (max-absolute / 127, nearest rounding, clipping [-127,127]). Native scaled FP16 outputs are followed by FP32 scale/bias and the recorded FP16 interface cast. 18 residual 3×3 convolutions use INT8; other layers/additions remain floating point. A natural FP32 suffix and the actual INT8-led suffix are separate paths.

The margin reanalysis is retrospective on these existing records. Before this reanalysis, thresholds [0,.01,.025,.05,.1,.2,.5,.9] were fixed. No calibration, new inference or independent confirmation was performed. The ratio |L|/T_hat uses the original high-precision reference and downstream Jacobian; it is not a low-cost online selector. All thresholds, discarded cases and retained negatives are reported.

Statistical intervals use 2000 paired image-cluster bootstrap draws, class-stratified, seed 92501; all interfaces/modes of an image stay together. Results condition on one trained checkpoint, not a distribution of training seeds. Timing groups are not independent-image confidence intervals.
