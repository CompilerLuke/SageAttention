# SageAttention4/SageAttention3 Ratio Grid Notes

Default grid:
- sequence lengths: 8k, 16k, 32k, 64k
- batch sizes: 1, 2, 4, 8, 16, 32, 64
- heads: 16
- kernels: sageattn3 and sageattn4 only

Known skipped configurations:
- 32k, B=64: S3 preprocess CUBLAS_STATUS_ALLOC_FAILED
- 64k, B=32: S3 preprocess CUBLAS_STATUS_ALLOC_FAILED
- 64k, B=64: S3 preprocess CUDA OOM

128k is not part of the default grid because previous runs failed during TMA descriptor initialization before S3/S4 timing completed.
