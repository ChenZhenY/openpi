"""Script to verify cotraining data loader is working correctly."""

import logging
from collections import Counter

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

logging.basicConfig(level=logging.INFO)


def verify_data_loader(config_name: str, num_samples: int = 100):
    """Verify that the data loader works correctly for cotraining.
    
    Args:
        config_name: Name of the config to test
        num_samples: Number of samples to check
    """
    print(f"\n{'='*60}")
    print(f"Verifying config: {config_name}")
    print(f"{'='*60}\n")
    
    # Get config
    config = _config.get_config(config_name)
    print(f"Config loaded: {config.name}")
    
    # Create data config
    data_config = config.data.create(config.assets_dirs, config.model)
    print(f"\nData config created:")
    print(f"  - repo_id: {data_config.repo_id}")
    print(f"  - repo_ids: {data_config.repo_ids}")
    print(f"  - asset_id: {data_config.asset_id}")
    print(f"  - prompt_from_task: {data_config.prompt_from_task}")
    
    # Create dataset
    dataset = _data_loader.create_torch_dataset(
        data_config, 
        action_horizon=config.model.action_horizon,
        model_config=config.model
    )
    print(f"\nDataset created:")
    print(f"  - Type: {type(dataset).__name__}")
    print(f"  - Length: {len(dataset)}")
    
    # If it's a ConcatDataset, show individual dataset sizes
    from torch.utils.data import ConcatDataset
    if isinstance(dataset, ConcatDataset):
        print(f"  - Number of datasets: {len(dataset.datasets)}")
        for i, sub_dataset in enumerate(dataset.datasets):
            print(f"    Dataset {i}: length = {len(sub_dataset)}")
            # Try to get a sample to verify it works
            if len(sub_dataset) > 0:
                sample = sub_dataset[0]
                if isinstance(sample, dict) and "prompt" in sample:
                    prompt_preview = str(sample["prompt"])[:50] + "..." if len(str(sample["prompt"])) > 50 else str(sample["prompt"])
                    print(f"      Sample prompt: {prompt_preview}")
    
    # Apply transforms
    transformed_dataset = _data_loader.transform_dataset(dataset, data_config, skip_norm_stats=False)
    print(f"\nTransformed dataset:")
    print(f"  - Type: {type(transformed_dataset).__name__}")
    print(f"  - Length: {len(transformed_dataset)}")
    
    # Create data loader
    data_loader = _data_loader.create_data_loader(
        config,
        shuffle=True,
        num_batches=10,  # Just test a few batches
        framework="pytorch",
    )
    print(f"\nData loader created")
    
    # Test loading samples
    print(f"\nTesting data loading (checking {num_samples} samples)...")
    sample_count = 0
    prompt_counter = Counter()
    dataset_source_counter = Counter()  # Track which dataset samples come from
    
    # If it's a ConcatDataset, we can track which dataset each sample comes from
    from torch.utils.data import ConcatDataset
    dataset_lengths = None
    if isinstance(dataset, ConcatDataset):
        dataset_lengths = [len(d) for d in dataset.datasets]
        print(f"  Individual dataset lengths: {dataset_lengths}")
        if data_config.dataset_weights:
            print(f"  Dataset weights: {data_config.dataset_weights}")
    
    for batch_idx, (observation, actions) in enumerate(data_loader):
        batch_size = actions.shape[0]
        print(f"  Batch {batch_idx}: batch_size={batch_size}")
        
        # Check observation structure
        if hasattr(observation, 'images'):
            print(f"    Images keys: {list(observation.images.keys())}")
        if hasattr(observation, 'state'):
            print(f"    State shape: {observation.state.shape}")
        if hasattr(observation, 'prompt'):
            # Count unique prompts to see diversity
            prompts = observation.prompt
            if hasattr(prompts, 'numpy'):
                prompts = prompts.numpy()
            for prompt in prompts:
                prompt_str = str(prompt)[:30]  # Truncate for counting
                prompt_counter[prompt_str] += 1
        
        print(f"    Actions shape: {actions.shape}")
        
        sample_count += batch_size
        if sample_count >= num_samples:
            break
    
    print(f"\nLoaded {sample_count} samples total")
    print(f"Unique prompt prefixes: {len(prompt_counter)}")
    if len(prompt_counter) > 0:
        print(f"Top 5 most common prompts:")
        for prompt, count in prompt_counter.most_common(5):
            print(f"  {prompt}... : {count} samples")
    
    # If weighted sampling is used, provide some statistics
    if data_config.dataset_weights and dataset_lengths:
        print(f"\nWeighted Sampling Info:")
        print(f"  Dataset lengths: {dataset_lengths}")
        print(f"  Dataset weights: {list(data_config.dataset_weights)}")
        total_weight = sum(data_config.dataset_weights)
        normalized_weights = [w / total_weight for w in data_config.dataset_weights]
        print(f"  Normalized weights: {[f'{w:.3f}' for w in normalized_weights]}")
        print(f"  Expected sampling ratio (per sample): {normalized_weights}")
    
    print(f"\n{'='*60}")
    print("Verification complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python verify_cotraining_data_loader.py <config_name> [num_samples]")
        print("\nExample:")
        print("  python verify_cotraining_data_loader.py pi05_libero_cotraining_all_pairs_step_30_1107_pi_libero 100")
        sys.exit(1)
    
    config_name = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    verify_data_loader(config_name, num_samples)

