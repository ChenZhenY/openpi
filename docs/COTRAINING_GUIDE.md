# Cotraining Guide

This guide explains how to use the cotraining feature to train on multiple datasets simultaneously.

## Overview

The cotraining feature allows you to:
1. Combine multiple datasets for training
2. Share transforms and normalization stats across datasets
3. Use per-dataset prompt handling
4. Adjust sampling ratios between datasets

## Basic Usage

### 1. Create a Multi-Dataset Config

Use `MultiDatasetLiberoDataConfig` instead of `LeRobotLiberoDataConfig`:

```python
TrainConfig(
    name="my_cotraining_config",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10),
    data=MultiDatasetLiberoDataConfig(
        repo_ids=["dataset1", "dataset2", "dataset3"],
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
        assets=AssetsConfig(
            assets_dir="./assets/my_config",
            asset_id="shared_norm_stats",
        ),
    ),
    # ... rest of config
)
```

### 2. Verify Your Setup

Run the verification script to check that everything works:

```bash
python scripts/verify_cotraining_data_loader.py <config_name> [num_samples]
```

Example:
```bash
python scripts/verify_cotraining_data_loader.py pi05_libero_cotraining_all_pairs_step_30_1107_pi_libero 100
```

The script will:
- Show dataset information (sizes, types)
- Test data loading
- Display sample statistics
- Verify transforms are applied correctly

## Weighted Sampling

### Adjusting Sampling Ratios

By default, datasets are sampled uniformly. To adjust the sampling ratio, use `dataset_weights`:

```python
data=MultiDatasetLiberoDataConfig(
    repo_ids=["large_dataset", "small_dataset"],
    dataset_weights=[1.0, 2.0],  # small_dataset sampled 2x more often
    # ... rest of config
)
```

**Important Notes:**
- Weights are normalized automatically
- `[2.0, 1.0]` means dataset 0 is sampled twice as often as dataset 1
- Weights apply per-sample, not per-dataset
- Only works with PyTorch framework (`framework="pytorch"`)
- Requires `shuffle=True` in the data loader

### Example: Balancing Dataset Sizes

If you have:
- Dataset 1: 10,000 samples
- Dataset 2: 1,000 samples

To balance them (sample equally from each):
```python
dataset_weights=[1.0, 10.0]  # Dataset 2 gets 10x weight to compensate for size
```

To give Dataset 1 more samples overall:
```python
dataset_weights=[2.0, 1.0]  # Dataset 1 sampled 2x more often
```

## How It Works

### Data Flow

1. **Dataset Creation**: Each `repo_id` creates a separate `LeRobotDataset`
2. **Prompt Handling**: Each dataset uses its own task metadata if `prompt_from_task=True`
3. **Concatenation**: Datasets are combined using `ConcatDataset`
4. **Transforms**: All datasets share the same transforms (repack, data, model)
5. **Normalization**: All datasets use the same normalization stats (from `asset_id`)
6. **Sampling**: Uses `WeightedRandomSampler` if weights are provided, otherwise uniform

### Shared vs Per-Dataset

**Shared (same for all datasets):**
- Transforms (repack, data, model)
- Normalization stats (from `asset_id`)
- Action sequence keys
- Model configuration

**Per-Dataset:**
- Task prompts (if `prompt_from_task=True`)
- Dataset-specific metadata (FPS, etc.)

## Verification Checklist

When setting up cotraining, verify:

- [ ] All datasets have compatible features (same keys, shapes)
- [ ] Normalization stats are appropriate for all datasets
- [ ] Transforms work correctly for all datasets
- [ ] Sampling ratios are as expected (use verification script)
- [ ] Prompts are correctly loaded from each dataset

## Troubleshooting

### Error: "dataset_weights length must match repo_ids length"
- Ensure `dataset_weights` has the same length as `repo_ids`

### Error: "Normalization stats not found"
- Make sure `asset_id` points to valid normalization stats
- Run `scripts/compute_norm_stats.py` if needed

### Weighted sampling not working
- Check that `framework="pytorch"` (not JAX)
- Ensure `shuffle=True` in training config
- Verify `dataset_weights` is not None

### Datasets have different features
- All datasets must have the same keys after repack transforms
- Check that transforms are compatible across datasets

## Example Configs

See `pi05_libero_cotraining_all_pairs_step_30_1107_pi_libero` in `config.py` for a complete example.

