#!/usr/bin/env python3.6

from pathlib import Path

import click

from semrel.data.scripts.corpus import from_index_documents_gen
from semrel.data.scripts.maps import MapMaker
from semrel.data.scripts.utils.io import save_lines, save_tensor
from semrel.data.scripts.vectorizers import FastTextVectorizer


@click.command()
@click.option('--input-path', required=True, type=str,
              help='Path to corpora relation files list.')
@click.option('--model', required=True, type=str,
              metavar='model.bin',
              help="Paths to fasttext model.")
@click.option('--output-paths', required=True, type=(str, str),
              metavar='fasttext.map.keys fasttext.map.pt',
              help='Paths for saving keys and map files.')
def main(input_path, model, output_paths):
    vectorizer = FastTextVectorizer(model_path=model)
    mapmaker = MapMaker(vectorizer=vectorizer)

    documents = from_index_documents_gen(relations_files_index=Path(input_path))
    keys, vectors = mapmaker.make_map(documents)

    keys_path, vectors_path = output_paths
    save_lines(Path(f'{keys_path}'), keys)
    save_tensor(Path(f'{vectors_path}'), vectors)


if __name__ == '__main__':
    main()
