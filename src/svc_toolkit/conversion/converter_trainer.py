import os

from so_vits_svc_fork.preprocessing.preprocess_split import preprocess_split
from so_vits_svc_fork.preprocessing.preprocess_resample import preprocess_resample
from so_vits_svc_fork.preprocessing.preprocess_flist_config import preprocess_config
from so_vits_svc_fork.preprocessing.preprocess_hubert_f0 import preprocess_hubert_f0
from so_vits_svc_fork.train import train

class ConverterTrainerFactory:
    def __init__(self) -> None:
        pass

    def create(self):
        return ConverterTrainer()
    
class ConverterTrainer:
    def __init__(self):
        pass

    # split
    # input ds_raw_raw
    # output ds_raw

    # resam
    # input ds_raw
    # ouput ds/44k

    # config
    # input ds/44k
    # filelist filelists/44k
    # config configs/44k/config.json

    # hubert
    # input ds/44k
    # config configs/44k/config.json

    # train
    # config configs/44k/config.json
    # model logs/44k

    def preprocess(self, input_dir: str, output_dir: str, split: bool = False):
        tmp_dir = os.path.join(output_dir, 'tmp')

        raw_dir = input_dir
        dataset_44k_dir = os.path.join(tmp_dir, 'dataset_44k')
        filelist_dir = os.path.join(tmp_dir, 'filelists_44k')
        config_path = os.path.join(output_dir, 'config.json')

        filelist_train_path = os.path.join(filelist_dir, 'train.txt')
        filelist_val_path = os.path.join(filelist_dir, 'val.txt')
        filelist_test_path = os.path.join(filelist_dir, 'test.txt')

        if split:
            raw_dir = os.path.join(tmp_dir, 'dataset_raw')
            preprocess_split(
                input_dir=input_dir,
                output_dir=raw_dir,
                sr=44100,
            )

        preprocess_resample(
            input_dir=raw_dir,
            output_dir=dataset_44k_dir,
            sampling_rate=44100
        )

        preprocess_config(
            input_dir=dataset_44k_dir,
            train_list_path=filelist_train_path,
            val_list_path=filelist_val_path,
            test_list_path=filelist_test_path,
            config_path=config_path,
            config_name='so-vits-svc-4.0v1'
        )

        preprocess_hubert_f0(
            input_dir=dataset_44k_dir,
            config_path=config_path,
            f0_method='crepe'
        )

    def train(self, config_path: str, model_path: str):
        train(
            config_path=config_path,
            model_path=model_path
        )