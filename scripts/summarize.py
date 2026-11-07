"""Summarize finished runs: mean and std of the final target normalized score over seeds.
Both the population std (ddof=0, used for the MuJoCo tables) and the sample std
(ddof=1, used for the AntMaze and Adroit tables) are printed.

Usage:
    python scripts/summarize.py --dir ./runs
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='./runs')
    args = parser.parse_args()

    scores = defaultdict(dict)
    for final in sorted(Path(args.dir).glob('*/seed*/final.json')):
        task, seed = final.parent.parent.name, final.parent.name
        with open(final) as f:
            scores[task][seed] = json.load(f)['test/target normalized score']

    print(f"{'task':45s} {'seeds':>5s} {'mean':>8s} {'std0':>7s} {'std1':>7s}")
    for task, by_seed in sorted(scores.items()):
        values = np.array(list(by_seed.values()))
        std1 = values.std(ddof=1) if len(values) > 1 else float('nan')
        print(f"{task:45s} {len(values):5d} {values.mean():8.2f} {values.std():7.2f} {std1:7.2f}")


if __name__ == '__main__':
    main()
