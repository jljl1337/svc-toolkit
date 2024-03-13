import os
from argparse import ArgumentParser

import torch
import matplotlib.pyplot as plt

from separation.evaluator import Evaluator

def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str, required=True)
    parser.add_argument('-t', '--test_csv', type=str, required=True)
    parser.add_argument('-e', '--experiment', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, default='./evaluation')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    evaluator = Evaluator(device)
    df_result = evaluator.evaluate(args.model_dir, args.test_csv)

    df_result.boxplot(grid=False)

    experiment_dir = os.path.join(args.output_dir, args.experiment)
    os.makedirs(experiment_dir, exist_ok=True)

    plt.savefig(os.path.join(experiment_dir, 'boxplot.png'))
    df_result.to_csv(os.path.join(experiment_dir, 'result.csv'), index=False)

    summary_df = evaluator.summary(df_result)
    summary_df.to_csv(os.path.join(experiment_dir, 'summary.csv'))


if __name__ == '__main__':
    main()