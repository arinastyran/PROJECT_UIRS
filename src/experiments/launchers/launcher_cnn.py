"""Main launcher for CNN experiments with parallel execution and progress tracking.

Examples
--------
Run from command line:
    $ python launcher_cnn.py --config-file cnn_cv.yaml --script-name cnn_cv.py
"""

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from utils.training import (
    ProgressTracker,
    load_config,
    extract_experiment_name,
    run_training_instance,
)


def main():
    """Main function to run multiple CNN training instances in parallel."""

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
    progress_dir = output_dir.joinpath('progress')
    script_path = Path('src/experiments').joinpath(experiment_type, args.script_name)

    # Get experiment names.
    experiment_names = [
        extract_experiment_name(experiment['experiment_params']) for experiment in experiments
    ]

    # Initialize progress tracker.
    progress_tracker = ProgressTracker(progress_dir, experiment_names)

    try:
        progress_tracker.start_monitoring()

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
                output_dir,
                progress_dir
            ): experiment_name for experiment_name in experiment_names}

            results = []
            for future in as_completed(future_to_config):
                result = future.result()
                results.append(result)

                # Update progress file to mark as completed/failed.
                exp_name = result['experiment_name']
                progress_file = progress_dir.joinpath(f'{exp_name}_progress.json')

                if progress_file.exists():
                    with open(progress_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    data['status'] = 'completed' if result['success'] else 'failed'
                    data['current_step'] = data['total_steps']

                    with open(progress_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f)

        # Stop progress monitoring.
        progress_tracker.stop_monitoring_func()

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

    finally:
        # Clean up progress directory
        progress_tracker.stop_monitoring_func()
        if progress_dir.exists():
            try:
                shutil.rmtree(progress_dir)
                print(f"Cleaned up progress directory: {progress_dir}.")
            except (OSError, PermissionError) as e:
                print(f"Warning: Could not clean up progress directory {progress_dir}: {e}.")


if __name__ == "__main__":
    sys.exit(main())
