1. Launch policy server: `uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_mfm --policy.dir=/media/robot/Data_2/openpi_ckpt/pi05_mfm/pi05_MFM_delta_1209_2/4000`
2. Run rollout code: using `atm-pipeline` conda env (we install openpi-client in it, and it has the gello and RL2 env steup)
   1. 