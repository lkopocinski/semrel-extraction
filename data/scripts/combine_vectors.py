#!/usr/bin/env python3

import argparse
from pathlib import Path

import argcomplete
import torch
import torch.nn as nn


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-in', required=True, help='File with relations.')
    parser.add_argument('--output-path', required=True, help='Directory to save vectors.')
    parser.add_argument('--elmo-map', required=True, nargs=2, metavar='elmo.map.pt elmo.map.keys', help="Elmo vectors and keys files.")
    parser.add_argument('--fasttext-map', required=True, nargs=2, metavar='fasttext.map.pt fasttext.map.keys', help="Fasttext vectors and keys files.")
    parser.add_argument('--retrofit-map', required=True, nargs=2, metavar='retrofit.map.pt retrofit.map.keys', help="Retrofit vectors and keys files.")

    argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def file_rows(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            yield line.strip().split('\t')


def load_map(vectors_file, keys_file):
    with open(vectors_file, 'r', encoding='utf-8') as f:
        for line in f



def max_pool(tensor):
    pool = nn.MaxPool1d(5, stride=0)
    tensor = tensor.transpose(2, 1)
    output = pool(tensor)
    return output.transpose(2, 1).squeeze()


def get_tensor(doc_id, sent_id, token_indices, vec_map, vec_size=1024):
    vectors = [vec_map[(doc_id, sent_id, idx)] for idx in token_indices]
    vectors = torch.FloatTensor(vectors).unsqueeze(0)
    tensor = torch.zeros(1, 5, vec_size)
    tensor[:, 0:vectors.shape[1], :] = vectors
    return tensor


def make_tensors_map(path: Path, vec_map, vec_size):
    rel_map = {}
    for row in file_rows(path):
        cat_id = row[0]
        label = row[1]
        doc_id = row[2]

        channel1 = row[5]
        channel2 = row[11]

        sent_id1 = row[3]
        sent_id2 = row[9]

        lemma1 = row[4]
        lemma2 = row[10]

        tokens1 = eval(row[7])
        tokens2 = eval(row[13])

        if len(tokens1) > 5 or len(tokens2) > 5:
            continue

        rel_key = (cat_id, label, doc_id, sent_id1, sent_id2, channel1, channel2, row[7], row[13], lemma1, lemma2)
        rel_map[rel_key] = (
            get_tensor(doc_id, sent_id1, tokens1, vec_map, vec_size),
            get_tensor(doc_id, sent_id2, tokens2, vec_map, vec_size)
        )

    return rel_map


def main(argv=None):
    args = get_args(argv)
    elmo_map = load_map(*args.elmo_map)
    fasttext_map = load_map(*args.fasttext_map)
    retrofit_map = load_map(*args.retrofit_map)



    source_path = Path(f'{args.data_in}/relations.112.114.115.context.uniq')
    for vec_map, vec_size, save_name in [(elmo_map, 1024, 'elmo.rel.pt'), (fasttext_map, 300, 'fasttext.rel.pt'), (retrofit_map, 300, 'retrofit.rel.pt')]:
        rel_map = make_tensors_map(source_path, vec_map, vec_size)
        keys, vec = zip(*rel_map.items())
        vec1, vec2 = zip(*vec)
        output1 = max_pool(torch.cat(vec1))
        output2 = max_pool(torch.cat(vec2))

        concat_dump = torch.cat([output1, output2], dim=1)
        torch.save(concat_dump, save_name)

        with open(f'{save_name}.keys', 'w', encoding='utf-8') as f:
            for key in keys:
                f.write(f'{key}\n')


if __name__ == '__main__':
    main()

