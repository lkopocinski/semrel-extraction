#!/usr/bin/env python3

import argparse
import os
from itertools import permutations, product
import glob
from pathlib import Path
from relation import Relation

import argcomplete
from utils import corpora_files, load_document, id_to_sent_dict, \
    is_ner_relation, is_in_channel, get_relation_element, \
    get_nouns_idx, get_lemma


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-in', required=True, help='Directory with split lists of files.')
    parser.add_argument('--output-path', required=True, help='Directory to save generated datasets.')

    argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def main(argv=None):
    args = get_args(argv)
    for set_name in ['train', 'valid', 'test']:
        source_dir = os.path.join(args.data_in, set_name)
        for list_file in glob.glob(f'{source_dir}/*.list'):
            file_path = os.path.join(args.output_path, 'negative')
            file_name = f'{get_file_name(list_file)}.context'
            lines = generate(list_file, ('BRAND_NAME', 'PRODUCT_NAME'))
            save_lines(file_path, file_name, lines)


def save_lines(path, file_name, lines):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f'List saving filed. Can not create {path} directory.')
    else:
        file_path = os.path.join(path, file_name)
        with open(file_path, 'w', encoding='utf-8') as out_file:
            for line in lines:
                out_file.write(f'{line}\n')


def get_file_name(file_path):
    return Path(file_path).stem


def generate(list_file, channels):
    for corpora_file, relations_file in corpora_files(list_file):
        document = load_document(corpora_file, relations_file)
        sentences = id_to_sent_dict(document)

        relations = {}
        relidxs = {}
        for relation in document.relations():
            if is_ner_relation(relation):
                if is_in_channel(relation, channels):
                    f = relation.rel_from()
                    t = relation.rel_to()
                    f_sent_id = f.sentence_id()
                    t_sent_id = t.sentence_id()

                    f_element = get_relation_element(f, sentences)
                    t_element = get_relation_element(t, sentences)
                    f_indices = tuple(f_element.indices)
                    t_indices = tuple(t_element.indices)

                    relations[((f_sent_id, f_indices), (t_sent_id, t_indices))] = (relation, f_element.context, t_element.context)
                    relations[((t_sent_id, t_indices), (f_sent_id, f_indices))] = (relation, t_element.context, f_element.context)

                    for f_idx in f_indices:
                        relidxs[(f_sent_id, f_idx)] = (f_indices, f_element.channel)
                    for t_idx in t_indices:
                        relidxs[(t_sent_id, t_idx)] = (t_indices, t_element.channel)

        for rel, rel_value in relations.items():
            relation, f_context, t_context = rel_value
            ((f_sent_id, f_indices), (t_sent_id, t_indices)) = rel

            f_nouns = get_nouns_idx(sentences[f_sent_id])
            t_nouns = get_nouns_idx(sentences[t_sent_id])

            if f_sent_id == t_sent_id:
                generator = permutations(f_nouns, 2)
            else:
                generator = product(f_nouns, t_nouns)

            for f_idx, t_idx in generator:
                try:
                    _f_idxs, _f_channel_name = relidxs[(f_sent_id, f_idx)]
                except KeyError:
                    _f_idxs = None
                    _f_channel_name = ''
                    pass

                try:
                    _t_idxs, _t_channel_name = relidxs[(t_sent_id, t_idx)]
                except KeyError:
                    _t_idxs = None
                    _t_channel_name = ''
                    pass

                if _t_idxs and _f_idxs:
                    if ((t_sent_id, _t_idxs), (f_sent_id, _f_idxs)) in relations:
                        continue

                f_lemma = get_lemma(sentences[f_sent_id], f_idx)
                t_lemma = get_lemma(sentences[t_sent_id], t_idx)
                source = Relation.Element(f_lemma, _f_channel_name, [f_idx], f_context)
                target = Relation.Element(t_lemma, _t_channel_name, [t_idx], t_context)
                yield Relation(source, target)


if __name__ == "__main__":
    main()
