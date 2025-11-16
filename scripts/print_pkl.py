import pickle
import torch
import jax

with open('/srv/rl2-lab/flash8/rbansal66/openpi_rollout/openpi/save_data/output_actions_float32_save.pkl', 'rb') as f:
    output_actions = pickle.load(f)

print(len(output_actions))