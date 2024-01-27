import os

import torch
import pandas as pd
import matplotlib.pyplot as plt
from torchmetrics.audio import SignalDistortionRatio, ScaleInvariantSignalDistortionRatio
from dotenv import load_dotenv
from tqdm import tqdm

from audio import load
from separator import Separator

class Evaluator:
    def __init__(self, device):
        self.device = device
        self.sdr = SignalDistortionRatio()
        self.sisdr = ScaleInvariantSignalDistortionRatio()

    def _flatten_tensor(self, wave):
        wave = wave.reshape(-1)
        return torch.from_numpy(wave)

    def evaluate(self, model_path, test_csv, window_length, hop_length, patch_length):
        separator = Separator(model_path, self.device)
        df_test = pd.read_csv(test_csv)
        df_result = pd.DataFrame(columns=['song', 'sdr', 'sisdr', 'nsdr', 'nsisdr'])

        for _index, row in tqdm(df_test.iterrows(), total=len(df_test)):
            mixture_path = row['mixture_path']
            stem_path = row['stem_path']
            mixture_wave, _ = load(mixture_path)
            stem_wave, _ = load(stem_path)
            mixture_tensor = self._flatten_tensor(mixture_wave)
            stem_tensor = self._flatten_tensor(stem_wave)

            estimate_wave = separator.separate(mixture_wave, window_length, hop_length, patch_length)
            estimate_tensor = self._flatten_tensor(estimate_wave)

            sdr_num = float(self.sdr(estimate_tensor, stem_tensor))
            sisdr_num = float(self.sisdr(estimate_tensor, stem_tensor))
            nsdr_num = sdr_num - float(self.sdr(mixture_tensor, stem_tensor))
            nsisdr_num = sisdr_num - float(self.sisdr(mixture_tensor, stem_tensor))
            
            df_result.loc[len(df_result)] = [row['song'], sdr_num, sisdr_num, nsdr_num, nsisdr_num]

        return df_result

def main():
    load_dotenv(override=True)
    WIN_LENGTH = int(os.getenv('WIN_LENGTH'))
    HOP_LENGTH = int(os.getenv('HOP_LENGTH'))
    PATCH_LENGTH = int(os.getenv('PATCH_LENGTH'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    evaluator = Evaluator(device)
    df_result = evaluator.evaluate('model/test4096_nozero/20240127_054904/best-epoch=92.ckpt', './musdb_test.csv', WIN_LENGTH, HOP_LENGTH, PATCH_LENGTH)
    # Create a box and whisker plot for each column
    df_result.boxplot(grid=False)
    # Save the figure
    plt.savefig('boxplot0.png')
    df_result.to_csv('result0.csv', index=False)


if __name__ == "__main__":
    main()