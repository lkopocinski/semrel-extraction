#!/usr/bin/env python3

import argparse
from pathlib import Path

import argcomplete

from utils.io import save_lines
from sampler import sample_positive, sample_negative


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-in', required=True, help='Directory with context format files.')
    parser.add_argument('--output-path', required=True, help='Directory for saving sampled datasets.')
    parser.add_argument('--train-size', nargs=2, type=int, required=True,
                        help='Train dataset batch sizes [positive, negative]')
    parser.add_argument('--valid-size', nargs=2, type=int, required=True,
                        help='Validation dataset batch sizes [positive, negative]')
    parser.add_argument('--test-size', nargs=2, type=int, required=True,
                        help='Test dataset batch sizes [positive, negative]')

    argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def main(argv=None):
    args = get_args(argv)

    for set_name, size in zip(['train', 'valid', 'test'], [args.train_size, args.valid_size, args.test_size]):
        for label_type, sample, batch_size in zip(['positive', 'negative'], [sample_positive, sample_negative], size):
            source_path = Path(f'{args.data_in}/{set_name}/{label_type}')
            if source_path.is_dir():
                for file_path in source_path.glob('*.context'):
                    out_file_path = Path(f'{args.output_path}/{set_name}/{label_type}/{file_path.stem}.sampled')
                    lines = sample(file_path, batch_size)
                    save_lines(out_file_path, lines)


if __name__ == '__main__':
    main()
