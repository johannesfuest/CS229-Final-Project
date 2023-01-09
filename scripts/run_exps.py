### CS229 Code Starts Here ###

import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument('ds_name', type=str)
parser.add_argument('to_impute', type=str)
args = parser.parse_args()

ds_name = args.ds_name
to_impute = args.to_impute

exp_dir = Path(f"exp/{ds_name}/ddpm_cb_best/config.toml")
pipeline = Path(f"scripts/pipeline.py")

props_missing_data = [0.01, 0.1, 0.3, 0.5]
types = ["MCAR", "MNAR", "MAR"]

# Runs a total of twelve different experiments, with comparison turned on
for type in types:
    for prop in props_missing_data:
        subprocess.run(['python', f'{pipeline}', '--config', f'{exp_dir}', '--sample_partial',
        '--to_impute', f'{to_impute}', '--exp_type', f'{type}', '--exp_prop', f'{prop}', '--compare'], check=True)

### CS229 Code Ends Here ###
