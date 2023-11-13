import os
from numpy import random
from torch.utils.data import Dataset

# from separation.audio import duration
from audio import duration, load, pre_stft, stft
import contants

class MusicDataset(Dataset):
    def __init__(self, root) -> None:
        self.chunk_length = 20
        self.margin_length = 0.5

        self.root = root
        self.chunk_list = []

        for dir in os.listdir(root):
            vocal_path = os.path.join(root, dir, 'vocals.wav')
            accompaniment_path = os.path.join(root, dir, 'accompaniment.wav')
            mixture_path = os.path.join(root, dir, 'mixture.wav')
            length = duration(mixture_path)
            n_chunks = int((length - self.margin_length) // self.chunk_length)
            for i in range(n_chunks):
                start = i * (self.chunk_length - self.margin_length)
                self.chunk_list.append((mixture_path, vocal_path,
                                        accompaniment_path, start,
                                        self.chunk_length))
        
    def __len__(self):
        return len(self.chunk_list)

    def __getitem__(self, index):
        chunk_item = self.chunk_list[index]
        path_list = chunk_item[0:3]
        start, length = chunk_item[3:5]
        post_stft_list = []
        for path in path_list:
            audio_data, sample_rate = load(path, offset=start, duration=length)
            audio_data = pre_stft(audio_data, sample_rate)
            post_stft = stft(audio_data)
            post_stft = abs(post_stft)
            post_stft = post_stft.transpose(2, 1, 0)
            post_stft_list.append(post_stft)

        start = random.randint(low=1,
                               high=(post_stft_list[0].shape[2] - contants.T))
        for i in range(len(post_stft_list)):
            post_stft_list[i] = post_stft_list[i][:, :contants.F, start: start + contants.T]

        return (*post_stft_list, )



if __name__ == "__main__":
    print('test')
    dataset = MusicDataset('/home/jljl1337/dataset/musdb18wav/train/')
    print(dataset.__getitem__(0)[0].shape)
    print(dataset.__getitem__(0)[1].shape)
    print(dataset.__getitem__(0)[2].shape)