import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv

import constants
from model import UNet
from audio import load, pre_stft, stft, save

class Separator():
    def __init__(self, model_dir, device) -> None:
        self.model_list = nn.ModuleList()
        self.device = device
        for i in range(2):
            checkpoint = torch.load(f'{model_dir}/net_{i}_099.pth')
            net = UNet()
            net.load_state_dict(checkpoint['model_state_dict'])
            net.eval()
            net.to(device)
            self.model_list.append(net)

    def split(self, data, T):
        original_size = data.size(3)
        new_size = math.ceil(original_size / T) * T
        data = F.pad(data, (0, new_size - original_size))
        return torch.cat(torch.split(data, T, dim=3), dim=0)

    def separate(self, mixture, source_sr):
        pre_stft_data = pre_stft(mixture, source_sr, constants.SAMPLE_RATE)
        post_stft_data = stft(pre_stft_data, constants.FRAME_LENGTH, constants.FRAME_STEP)
        post_stft_data = torch.from_numpy(post_stft_data)
        post_stft_data = post_stft_data[:, : constants.F, :]
        post_stft_data = abs(post_stft_data)
        post_stft_data = post_stft_data.unsqueeze(0).permute([0, 3, 2, 1])

        stft_length = post_stft_data.shape[3] # for istft

        post_stft_data = self.split(post_stft_data, constants.T)
        post_stft_data = post_stft_data.transpose(2, 3)
        post_stft_data = post_stft_data.to(self.device)

        wave_list = []

        for model in self.model_list:
            post_model_data = model(post_stft_data)
            post_model_data = post_model_data.transpose(2, 3)
            post_model_data = torch.cat(torch.split(post_model_data, 1, dim=0),
                                        dim=3)
            post_model_data = post_model_data.squeeze(0)[:, :, : stft_length]
            post_model_data = post_model_data.permute([2, 1, 0])
            post_model_data = post_model_data.cpu().detach().numpy()

            post_model_data = np.pad(post_model_data, ((0, 0), (0, 1025), (0, 0)),
                                    'constant')
            wave = stft(post_model_data, constants.FRAME_LENGTH,
                        constants.FRAME_STEP, inverse=True,
                        length=pre_stft_data.shape[0])
            wave_list.append(wave)
        
        return wave_list

    def separate_from_file(self, input_path, output_dir):
        audio_data, sample_rate = load(input_path)
        audio_data = torch.from_numpy(audio_data)

        separated_wave_list = self.separate(audio_data, sample_rate)

        os.makedirs(output_dir, exist_ok=True)
    
        for i in range(len(separated_wave_list)):
            save(f'{output_dir}/{constants.STEM[i]}_estimate.wav',
                 separated_wave_list[i],
                 constants.SAMPLE_RATE)

def main():
    load_dotenv()
    MODEL_DIR = os.getenv('MODEL_DIR')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    separator = Separator(MODEL_DIR, device)
    separator.separate_from_file('/home/jljl1337/dataset/musdb18wav/test/Forkupines - Semantics/mixture.wav', './output/')

if __name__ == "__main__":
    main()