# PTDB-1 full development shards

Each subdirectory is a self-contained private Kaggle script kernel. Submit with
an explicit T4 because Kaggle's 2026-07-18 P100/PyTorch image was incompatible:

```powershell
python -m kaggle kernels push -p PATH_TO_SHARD --accelerator NvidiaTeslaT4
```

After download, validate dataset shards with:

```powershell
python benchmarks/ptdb_v1/kaggle/full/validate_shard.py OUTPUT_DIRECTORY cifar10
```

Architecture-specific SVHN shards include the model argument, for example:

```powershell
python benchmarks/ptdb_v1/kaggle/full/validate_shard.py OUTPUT_DIRECTORY svhn resnet18
```

The protected holdout is not implemented or mounted in these kernels.
