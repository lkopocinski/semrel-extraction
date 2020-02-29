#!/usr/bin/env python3.6
from pathlib import Path

import click

from data.scripts.maps import MapMaker
from data.scripts.utils.corpus import from_index_documents_gen
from data.scripts.utils.io import save_lines, save_tensor
from data.scripts.utils.vectorizers import ElmoVectorizer, ElmoEmbedderVectorizer


@click.command()
@click.option('--input-path', required=True, type=str,
              help='Path to corpora relation files list.')
@click.option('--model', required=True, type=(str, str),
              metavar='options.json weights.hdf5',
              help="Paths to elmo model options and weights.")
@click.option('--output-paths', required=True, type=(str, str),
              metavar='elmo.map.keys elmo.map.pt',
              help='Paths for saving keys and map files.')
def main(input_path, model, output_paths):
    elmo_options, elmo_weights = model
    vectorizer = ElmoEmbedderVectorizer(options_path=elmo_options, weights_path=elmo_weights)
    mapmaker = MapMaker(vectorizer=vectorizer)

    documents = from_index_documents_gen(relations_files_index=Path(input_path))
    keys, vectors = mapmaker.make_map(documents)

    keys_path, vectors_path = output_paths
    save_lines(Path(f'{keys_path}'), keys)
    save_tensor(Path(f'{vectors_path}'), vectors)


if __name__ == '__main__':
    main()
