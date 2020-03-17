#!/usr/bin/env python3.6

from pathlib import Path

import click

from .maps import MapLoader
from .relations import RelationsLoader, RelationsEmbedder
from .utils.io import save_lines, save_tensor


@click.command()
@click.option('--input-path', required=True, type=str,
              help='Path to relations file.')
@click.option('--elmo-map', required=True, type=(str, str),
              metavar='elmo.map.keys elmo.map.pt',
              help="Elmo keys and vectors files.")
@click.option('--fasttext-map', required=True, type=(str, str),
              metavar='fasttext.map.keys fasttext.map.pt',
              help="Fasttext keys and vectors files.")
@click.option('--retrofit-map', required=True, type=(str, str),
              metavar='retrofit.map.keys retrofit.map.pt',
              help="Retrofit keys and vectors files.")
@click.option('--output-dir', required=True, type=str,
              help='Directory for saving relations embeddings.')
def main(input_path, elmo_map, fasttext_map, retrofit_map, output_dir):
    map_loaders = {
        'elmo': MapLoader(*elmo_map),
        'fasttext': MapLoader(*fasttext_map),
        'retrofit': MapLoader(*retrofit_map)
    }
    relations_path = Path(input_path)

    for name, load_map in map_loaders.items():
        vectors_map = load_map()
        relations_loader = RelationsLoader(relations_path)
        relations_embedder = RelationsEmbedder(relations_loader, vectors_map)

        relations_keys, relations_vectors = relations_embedder.embed()

        save_lines(Path(f'{output_dir}/{name}.rel.keys'), relations_keys)
        save_tensor(Path(f'{output_dir}/{name}.rel.pt'), relations_vectors)


if __name__ == '__main__':
    main()
