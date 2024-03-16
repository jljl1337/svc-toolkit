import os
import argparse
from argparse import ArgumentParser

import torch
import matplotlib.pyplot as plt

from separation.evaluator import Evaluator

def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str, required=True)
    parser.add_argument('-t', '--test_csv', type=str, required=True)
    parser.add_argument('-l', '--last', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    evaluator = Evaluator(device)
    df_result = evaluator.evaluate(args.model_dir, args.test_csv, args.last)

    df_result.boxplot(grid=False)

    plt.savefig(os.path.join(args.model_dir, f'boxplot{"_last" if args.last else ""}.png'))
    df_result.to_csv(os.path.join(args.model_dir, f'result{"_last" if args.last else ""}.csv'), index=False)

    summary_df = evaluator.summary(df_result)
    summary_df.to_csv(os.path.join(args.model_dir, f'summary{"_last" if args.last else ""}.csv'))


if __name__ == '__main__':
    main()