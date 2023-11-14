import os 
import torch
import numpy as np
import pandas as pd
from torchmetrics.audio import SignalDistortionRatio, ScaleInvariantSignalDistortionRatio
from dotenv import load_dotenv
from tqdm import tqdm

from audio import load
from separator import Separator

def main():
    load_dotenv()
    MODEL_DIR = os.getenv('MODEL_DIR')
    TEST_DIR = os.getenv('TEST_DIR')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    separator = Separator(MODEL_DIR, device)

    df = pd.DataFrame(columns=['song', 'stem', 'sdr'])
    sdr = SignalDistortionRatio()
    sdr = ScaleInvariantSignalDistortionRatio()

    for song in tqdm(os.listdir(TEST_DIR)):
        # separator.separate_from_file(f'{TEST_DIR}/{song}/mixture.wav', f'{TEST_DIR}/{song}')
        mixture, _ = load(f'{TEST_DIR}/{song}/mixture.wav')
        for stem in ['vocals', 'accompaniment']:
            estimate, _ = load(f'{TEST_DIR}/{song}/{stem}_estimate.wav')
            truth, _ = load(f'{TEST_DIR}/{song}/{stem}.wav')
            estimate = estimate.reshape(-1)
            truth = truth.reshape(-1)
            sdr_num = float(sdr(torch.from_numpy(estimate), torch.from_numpy(truth)))
            df.loc[len(df)] = [song, stem, sdr_num]

    df.to_csv('sisdr.csv', index=False)


if __name__ == "__main__":
    main()