from pathlib import Path
from typing import List

import click

from data_loader import BrandProductDataset, DatasetGenerator
from io import save_json
from runs import RUNS


def get_indices(data_dir: str,
                keys_file: str,
                vectors_files: List[str],
                balanced: bool = False,
                lexical_split: bool = False,
                in_domain: str = None,
                random_seed: int = 42):
    keys_path = f'{data_dir}/{keys_file}'
    vectors_files = [f'{data_dir}/{file}' for file in vectors_files]

    dataset = BrandProductDataset(keys_file=keys_path, vectors_files=vectors_files)
    ds_generator = DatasetGenerator(dataset, random_seed)
    return ds_generator.generate_datasets(balanced, lexical_split, in_domain)


@click.command()
@click.option('--data-dir', required=True, type=str)
@click.option('--data-keys', required=True, type=str)
@click.option('--runs-name', required=True, type=str)
def main(data_dir, data_keys, runs_name):
    indices = {}

    runs = RUNS[runs_name]
    for index, params in runs.items():
        in_domain = params.get('in_domain')
        lexical_split = params.get('lexical_split', False)
        methods = params.get('methods', [])

        train, valid, test = get_indices(
            data_dir=data_dir,
            keys_file=data_keys,
            vectors_files=[f'{method}.rel.pt' for method in methods],
            balanced=True,
            lexical_split=lexical_split,
            in_domain=in_domain,
        )

        indices[index] = {
            'train': train,
            'valid': valid,
            'test': test
        }

        save_json(indices, Path(f'./runs_indices.json'))


if __name__ == '__main__':
    main()
