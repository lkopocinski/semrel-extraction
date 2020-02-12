#!/usr/bin/env python3.6
from pathlib import Path

import click

import data.scripts.utils.vectorizers as vec
from data.scripts.maps import MapMaker
from data.scripts.utils.corpus import relations_file_paths, relations_documents_gen
from data.scripts.utils.io import save_lines, save_tensor


@click.command()
@click.option('--input-path', required=True, type=str,
              help='Path to corpora files.')
@click.option('--directories', required=True, nargs=1,
              help='Directories names with corpus files.')
@click.option('--elmo-model', required=True, type=(str, str),
              help="A path to elmo model options, weights.")
@click.option('--fasttext-model', required=True, type=str,
              help="A path to fasttext model.")
@click.option('--retrofit-model', required=True, type=str,
              help="File with retrofitted fasttext model.")
@click.option('--output-path', required=True, type=str,
              help='Directory for saving map files.')
def main(
        input_path,
        directories,
        elmo_model,
        fasttext_model,
        retrofit_model,
        output_path
):
    elmo_options, elmo_weights = elmo_model
    makers_dict = {
        'elmo': MapMaker(vectorizer=vec.ElmoVectorizer(elmo_options, elmo_weights)),
        'fasttext': MapMaker(vectorizer=vec.FastTextVectorizer(fasttext_model)),
        'retrofit': MapMaker(vectorizer=vec.RetrofitVectorizer(retrofit_model, fasttext_model))
    }
    relations_files = relations_file_paths(input_path, directories)

    for name, mapmaker in makers_dict.items():
        documents = relations_documents_gen(relations_files)
        keys, vectors = mapmaker.make_map(documents)

        save_lines(Path(f'{output_path}/{name}.map.keys'), keys)
        save_tensor(Path(f'{output_path}/{name}.map.pt'), vectors)


if __name__ == '__main__':
    main()
