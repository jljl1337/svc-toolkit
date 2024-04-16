import os
from argparse import ArgumentParser

from svc_toolkit.separation import constants
from svc_toolkit.separation.preprocess import moisesdb_mix, preprocess

def main() -> None:
    parser = ArgumentParser(description='Preprocess the dataset(s).')
    parser.add_argument('-o', '--csv_output_dir', type=str, default='./input_csv', help='CSV Output directory (default: ./input_csv)')
    parser.add_argument('-v', '--val_size', type=float, default=0.2, help='Validation size (default: 0.2)')
    parser.add_argument('-s', '--stem', type=str, default='vocals', help='Stem to preprocess (default: vocals)')
    parser.add_argument('-m', '--musdb_dir', type=str, required=True, help='Path to the MUSDB18 directory (required)')
    parser.add_argument('-M', '--moisesdb_dir', type=str, default='', help='Path to the MoisesDB directory (optional)')
    parser.add_argument('-w', '--moisesdb_wav_dir', type=str, required=True, help='Path to the MoisesDB wav directory (required)')
    args = parser.parse_args()

    if args.moisesdb_dir != '':
        moisesdb_mix(args.moisesdb_dir, args.moisesdb_wav_dir, args.stem)

    os.makedirs(args.output_dir, exist_ok=True)

    preprocess(args.musdb_dir, args.moisesdb_wav_dir, args.val_size, args.output_dir, args.stem, constants.SEED)

if __name__ == "__main__":
    main()