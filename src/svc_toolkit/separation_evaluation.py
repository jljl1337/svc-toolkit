import os
import argparse
from argparse import ArgumentParser

import torch
import matplotlib.pyplot as plt

from svc_toolkit.separation.evaluator import Evaluator

def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str, required=True)
    parser.add_argument('-t', '--test_csv', type=str, required=True)
    parser.add_argument('-p', '--precision', type=str, default='bf16')
    parser.add_argument('-l', '--last', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    evaluator = Evaluator(args.model_dir, device, args.precision, args.last)
    df_result = evaluator.evaluate(args.test_csv)

    boxplot_file_name = f'boxplot{"_last" if args.last else ""}.png'
    result_file_name = f'result{"_last" if args.last else ""}.csv'
    summary_file_name = f'summary{"_last" if args.last else ""}.csv'

    df_result.to_csv(os.path.join(args.model_dir, result_file_name), index=False)

    summary_df = evaluator.summary(df_result)
    summary_df.to_csv(os.path.join(args.model_dir, summary_file_name))

    df_result.boxplot(grid=False)
    plt.savefig(os.path.join(args.model_dir, boxplot_file_name))

if __name__ == '__main__':
    main()