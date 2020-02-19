#!/usr/bin/env python3.6
from pathlib import Path

import click
import torch

import sent2vec
from data.scripts.combine_vectors import RelationsLoader
from data.scripts.entities import Relation
from data.scripts.utils.corpus import from_index_documents_gen
from io import save_lines, save_tensor

PHRASE_LENGTH_LIMIT = 5


def make_sentence_map(relations_paths: Path):
    sentence_map = {}

    documents = from_index_documents_gen(relations_files_index=relations_paths)
    for document in documents:
        id_domain = document.directory
        id_document = document.id
        sentence_map[(id_domain, id_document)] = {}

        for sentence in document.sentences:
            sentence_index = int(sentence.id.replace('sent', ''))
            context = sentence.orths
            sentence_map[(id_domain, id_document)][sentence_index] = context

    return sentence_map


def make_vectors(relations_loader: RelationsLoader, sentence_map: dict):
    keys = []
    vectors = []

    for label, id_domain, relation in relations_loader.relations():
        id_document, member_from, member_to = relation

        if len(member_from.indices) > PHRASE_LENGTH_LIMIT or len(member_to.indices) > PHRASE_LENGTH_LIMIT:
            continue

        id_sentence_from = int(member_from.id_sentence.replace('sent', ''))
        id_sentence_to = int(member_to.id_sentence.replace('sent', ''))

        if id_sentence_from > id_sentence_to:
            # swap
            id_sentence_from = int(member_to.id_sentence.replace('sent', ''))
            id_sentence_to = int(member_from.id_sentence.replace('sent', ''))
            indices_from = member_to.indices
            indices_to = member_from.indices
        else:
            indices_from = member_from.indices
            indices_to = member_to.indices

        if id_sentence_from == id_sentence_to:
            context_left = sentence_map[(id_domain, id_document)].get(id_sentence_from - 1, [])
            context_between = mask_tokens(sentence_map[(id_domain, id_document)][id_sentence_from],
                                          indices_from + indices_to)
            context_right = sentence_map[(id_domain, id_document)].get(id_sentence_to + 1, [])
        elif (id_sentence_to - id_sentence_from) > 0:
            # be sure the indices are swapped
            context_between = []
            context_left = sentence_map[(id_domain, id_document)].get(id_sentence_from - 1, [])
            context_between.extend(mask_tokens(sentence_map[(id_domain, id_document)][id_sentence_from], indices_from))

            for i in range(id_sentence_from + 1, id_sentence_to):
                context_between.extend(sentence_map[(id_domain, id_document)].get(i, []))

            context_between.extend(mask_tokens(sentence_map[(id_domain, id_document)][id_sentence_to], indices_to))
            context_right = sentence_map[(id_domain, id_document)].get(id_sentence_to + 1, [])

        tokens = context_left + context_between + context_right
        sentence = ' '.join(tokens)

        keys.append(_make_key(label, id_domain, relation))
        vectors.append(torch.from_numpy(s2v.embed_sentence(sentence)))

    tensor = torch.cat(vectors)

    return keys, tensor


def _make_key(label: str, id_domain: str, relation: Relation):
    id_document, member_from, member_to = relation
    return '\t'.join([
        label, id_domain, id_document,
        member_from.id_sentence, member_from.channel, str(member_from.indices), member_from.lemma,
        member_to.id_sentence, member_to.channel, str(member_to.indices), member_to.lemma,
    ])


def mask_tokens(context, indices):
    return ['MASK' if index in indices else token for index, token in enumerate(context)]


@click.command()
@click.option('--relations-file', required=True, type=str,
              help='Path to relations file.')
@click.option('--documents-files', required=True, type=str,
              help='Path to relations file.')
@click.option('--model', required=True, type=str,
              metavar='model.bin',
              help="Paths to sent2vec model.")
@click.option('--output-paths', required=True, type=(str, str),
              metavar='sent2vec.map.keys sent2vec.map.pt',
              help='Paths for saving keys and map files.')
def main(relations_file, documents_files, model, output_paths):
    s2v = sent2vec.Sent2vecModel()
    s2v.load_model(model, inference_mode=True)

    sentence_map = make_sentence_map(Path(documents_files))
    relations_loader = RelationsLoader(relations_file)

    relations_keys, relations_vectors = make_vectors(relations_loader, sentence_map)

    keys_path, vectors_path = output_paths
    save_lines(Path(keys_path), relations_keys)
    save_tensor(Path(vectors_path), relations_vectors)


if __name__ == '__main__':
    main()
