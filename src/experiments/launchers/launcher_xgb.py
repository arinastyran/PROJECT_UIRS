"""Main launcher for XGB experiments.

Examples
--------
Run from command line:
    $ python launcher_xgb.py --config-file xgb_tuning.yaml --script-name xgb_tuning.py
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from utils.training import load_config, extract_experiment_name, run_training_instance


def main():
    """Main function to run multiple XGB training instances."""

    parser = argparse.ArgumentParser(description='Run multiple training instances.')
    parser.add_argument('--config-file',
                        type=str,
                        required=True,
                        help='YAML file containing all configurations.')
    parser.add_argument('--script-name',
                        type=str,
                        required=True,
                        help='Training script name.')
    parser.add_argument('--output-dir',
                        default='results/logs',
                        type=Path,
                        help='Base directory for experiment outputs.')
    parser.add_argument('--max-parallel',
                        default=9,
                        type=int,
                        help='Maximum number of parallel processes.')
    args = parser.parse_args()

    # Load configurations.
    experiment_model, experiment_type = args.config_file.removesuffix('.yaml').split('_')
    config_path = Path('configs').joinpath(args.config_file)
    config_params = load_config(config_path)
    experiments = config_params['experiments']
    print(f"Loaded {len(experiments)} configurations.")

    # Set up paths.
    output_dir = args.output_dir.joinpath(experiment_type, experiment_model)
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path('src/experiments').joinpath(experiment_type, args.script_name)

    # Get experiment names.
    experiment_names = [
        extract_experiment_name(experiment['experiment_params']) for experiment in experiments
    ]

    # Parallel execution
    print(f"Running {len(experiments)} experiments"
          f" with {args.max_parallel} parallel processes.")

    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        # Submit all jobs.
        experiments = config_params['experiments']
        future_to_config = {executor.submit(
            run_training_instance,
            script_path,
            args.config_file,
            experiment_name,
            output_dir
        ): experiment_name for experiment_name in experiment_names}

        results = []
        for future in as_completed(future_to_config):
            result = future.result()
            results.append(result)

    # Summary.
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful

    print("\n=== Summary ===")
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed experiments:")
        for result in results:
            if not result['success']:
                exp_name = result['config'].get('experiment_name', 'Unknown')
                print(f"  - {exp_name}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
