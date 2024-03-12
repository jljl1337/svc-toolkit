import torch
import pandas as pd
import matplotlib.pyplot as plt
from torchmetrics.audio import SignalDistortionRatio, ScaleInvariantSignalDistortionRatio
from tqdm import tqdm

from separator import Separator

class Evaluator:
    def __init__(self, device):
        self.device = device
        self.sdr = SignalDistortionRatio()
        self.sisdr = ScaleInvariantSignalDistortionRatio()

    def _flatten_tensor(self, wave):
        wave = wave.reshape(-1)
        return torch.from_numpy(wave)

    def evaluate(self, model_dir, test_csv):
        separator = Separator(model_dir, self.device)
        df_test = pd.read_csv(test_csv)
        df_result = pd.DataFrame(columns=['song', 'SDR', 'SI-SDR', 'NSDR', 'NSI-SDR'])

        for _index, row in tqdm(df_test.iterrows(), total=len(df_test)):
            mixture_path = row['mixture_path']
            stem_path = row['stem_path']
            mixture_wave, _ = separator.load_file(mixture_path)
            stem_wave, _ = separator.load_file(stem_path)
            mixture_tensor = self._flatten_tensor(mixture_wave)
            stem_tensor = self._flatten_tensor(stem_wave)

            estimate_wave, _ = separator.separate(mixture_wave)
            estimate_tensor = self._flatten_tensor(estimate_wave)

            sdr_num = float(self.sdr(estimate_tensor, stem_tensor))
            sisdr_num = float(self.sisdr(estimate_tensor, stem_tensor))
            nsdr_num = sdr_num - float(self.sdr(mixture_tensor, stem_tensor))
            nsisdr_num = sisdr_num - float(self.sisdr(mixture_tensor, stem_tensor))
            
            df_result.loc[len(df_result)] = [row['song'], sdr_num, sisdr_num, nsdr_num, nsisdr_num]

        return df_result

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    evaluator = Evaluator(device)
    df_result = evaluator.evaluate('model/all/20240202_064108', './musdb_test.csv')
    # Create a box and whisker plot for each column
    df_result.boxplot(grid=False)
    # Save the figure
    plt.savefig('boxplot_all_last_musdb.png')
    df_result.to_csv('result_all_last_musdb.csv', index=False)

    summary_df = pd.DataFrame(columns=['Mean', 'SD', 'Min', 'Max', 'Median'])
    for column in df_result.columns[1:]:
        summary_df.loc[column] = [
            df_result[column].mean(),
            df_result[column].std(),
            df_result[column].min(),
            df_result[column].max(),
            df_result[column].median()
        ]
    summary_df.to_csv('summary_all_last_musdb.csv')


if __name__ == "__main__":
    main()