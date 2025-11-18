import argparse
import jax

from openpi.policies import policy_config, libero_policy
from openpi.shared import download
from openpi.training import config as _config


def main(args):
    config = _config.get_config("pi05_libero")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")

    # Create a trained policy.
    policy = policy_config.create_trained_policy(config, checkpoint_dir, sample_kwargs={"num_steps": args.num_steps})
    # Run inference on a dummy example.
    example = libero_policy.make_libero_example()
    examples = [example] * args.batch_size

    print("Warming up...")
    outputs = policy.infer_batch(examples)
    jax.block_until_ready(outputs)
    print("Warmed up")

    print("Inferring...")
    outputs = policy.infer_batch(examples)
    jax.block_until_ready(outputs)
    output = outputs[0]
    print("actions", output["actions"].shape)
    print("policy_timing", output["policy_timing"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=10)
    args = parser.parse_args()
    main(args)
