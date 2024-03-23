import torch
import pandas as pd
from torchmetrics.audio import SignalDistortionRatio, ScaleInvariantSignalDistortionRatio
from tqdm import tqdm

from separation.separator import Separator
from separation.constants import CSV_MIXTURE_PATH_COLUMN, CSV_STEM_PATH_COLUMN

class Evaluator:
    def __init__(self, device):
        self.device = device
        self.sdr = SignalDistortionRatio()
        self.sisdr = ScaleInvariantSignalDistortionRatio()

    def _flatten_tensor(self, wave):
        wave = wave.reshape(-1)
        return torch.from_numpy(wave)

    def evaluate(self, model_dir, test_csv, last):
        separator = Separator(model_dir, self.device, last)
        df_test = pd.read_csv(test_csv)
        df_result = pd.DataFrame(columns=['song', 'SDR', 'SI-SDR', 'NSDR', 'NSI-SDR'])

        for _index, row in tqdm(df_test.iterrows(), total=len(df_test)):
            mixture_path = row[CSV_MIXTURE_PATH_COLUMN]
            stem_path = row[CSV_STEM_PATH_COLUMN]
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

    def summary(self, df: pd.DataFrame):
        summary_df = pd.DataFrame(columns=['Mean', 'SD', 'Min', 'Max', 'Median'])

        for column in df.columns[1:]:
            summary_df.loc[column] = [
                df[column].mean(),
                df[column].std(),
                df[column].min(),
                df[column].max(),
                df[column].median()
            ]

        return summary_df