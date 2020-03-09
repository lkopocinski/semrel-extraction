#!/usr/bin/env python3.6

from pathlib import Path

import click

from data.scripts.utils.io import save_json
from model.runs import RUNS
from model.scripts.utils.data_loader import BrandProductDataset, DatasetGenerator


def get_indices(keys_file: Path,
                balanced: bool = False,
                lexical_split: bool = False,
                in_domain: str = None,
                random_seed: int = 42):
    keys = BrandProductDataset._load_keys(keys_file)
    ds_generator = DatasetGenerator(keys, random_seed)
    return ds_generator.generate_datasets(balanced, lexical_split, in_domain)


@click.command()
@click.option('--dataset-keys', required=True, type=str)
@click.option('--output-path', required=True, type=str)
def main(dataset_keys, output_path):
    indices = {}


    spert_runs = RUNS['spert']
    for index, params in spert_runs.items():
        in_domain = params.get('in_domain')
        lexical_split = params.get('lexical_split', False)

        train, valid, test = get_indices(
            keys_file=Path(dataset_keys),
            balanced=True,
            lexical_split=lexical_split,
            in_domain=in_domain,
        )

        indices[index] = {
            'train': train,
            'valid': valid,
            'test': test
        }

    save_json(indices, Path(output_path))


if __name__ == '__main__':
    main()
