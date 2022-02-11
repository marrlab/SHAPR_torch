"""Run training with optional external parameters file."""

import argparse

from shapr import run_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--params',
        type=str,
        default=None,
        help='Config file'
    )

    args = parser.parse_args()
    run_train(params=args.params)
