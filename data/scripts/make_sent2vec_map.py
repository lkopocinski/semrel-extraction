#!/usr/bin/env python3.6
from pathlib import Path
from typing import List

import click
import torch

from data.scripts.entities import Relation, Member
from data.scripts.relations import RelationsLoader
from data.scripts.utils.corpus import from_index_documents_gen
from data.scripts.utils.io import save_lines, save_tensor
from data.scripts.utils.vectorizers import Sent2VecVectorizer


def make_sentence_map(relations_paths: Path) -> dict:
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


class RelationsMapMaker:
    PHRASE_LENGTH_LIMIT = 5
    MASK = 'MASK'

    def __init__(self, relations_loader: RelationsLoader, vectorizer: Sent2VecVectorizer):
        self.relations_loader = relations_loader
        self.vectorizer = vectorizer

    def _is_phrase_too_long(self, member: Member) -> bool:
        return len(member.indices) > self.PHRASE_LENGTH_LIMIT

    def _make_key(self, label: str, id_domain: str, relation: Relation):
        id_document, member_from, member_to = relation
        return '\t'.join([
            label, id_domain, id_document,
            member_from.id_sentence, member_from.channel, str(member_from.indices), member_from.lemma,
            member_to.id_sentence, member_to.channel, str(member_to.indices), member_to.lemma,
        ])

    def _mask_tokens(self, context: List[str], indices: tuple[int]):
        return [self.MASK
                if index in indices else token
                for index, token in enumerate(context)]

    def make_map(self, sentence_map: dict) -> [List, torch.tensor]:
        keys = []
        vectors = []

        for label, id_domain, relation in self.relations_loader.relations():
            id_document, member_from, member_to = relation

            if self._is_phrase_too_long(member_from) or self._is_phrase_too_long(member_to):
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
                context_between = self._mask_tokens(sentence_map[(id_domain, id_document)][id_sentence_from],
                                                    indices_from + indices_to)
                context_right = sentence_map[(id_domain, id_document)].get(id_sentence_to + 1, [])
            elif (id_sentence_to - id_sentence_from) > 0:
                # be sure the indices are swapped
                context_between = []
                context_left = sentence_map[(id_domain, id_document)].get(id_sentence_from - 1, [])
                context_between.extend(
                    self._mask_tokens(sentence_map[(id_domain, id_document)][id_sentence_from], indices_from))

                for i in range(id_sentence_from + 1, id_sentence_to):
                    context_between.extend(sentence_map[(id_domain, id_document)].get(i, []))

                context_between.extend(
                    self._mask_tokens(sentence_map[(id_domain, id_document)][id_sentence_to], indices_to))
                context_right = sentence_map[(id_domain, id_document)].get(id_sentence_to + 1, [])

            context = context_left + context_between + context_right

            key = self._make_key(label, id_domain, relation)
            vector = self.vectorizer.embed(context)

            keys.append(key)
            vectors.append(vector)

        tensor = torch.cat(vectors)

        return keys, tensor


@click.command()
@click.option('--relations-file', required=True, type=str,
              help='Path to relations file.')
@click.option('--documents-files', required=True, type=str,
              help='Path to corpora relation files list.')
@click.option('--model', required=True, type=str,
              metavar='model.bin',
              help="Paths to sent2vec model.")
@click.option('--output-paths', required=True, type=(str, str),
              metavar='sent2vec.map.keys sent2vec.map.pt',
              help='Paths for saving keys and map files.')
def main(relations_file, documents_files, model, output_paths):
    vectorizer = Sent2VecVectorizer(model_path=model)
    relations_loader = RelationsLoader(relations_file)
    sentence_map = make_sentence_map(relations_paths=Path(documents_files))

    mapmaker = RelationsMapMaker(relations_loader, vectorizer)
    relations_keys, relations_vectors = mapmaker.make_map(sentence_map)

    keys_path, vectors_path = output_paths
    save_lines(Path(keys_path), relations_keys)
    save_tensor(Path(vectors_path), relations_vectors)


if __name__ == '__main__':
    main()
