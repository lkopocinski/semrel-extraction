#!/usr/bin/env python3.6
from pathlib import Path
from typing import List, Tuple

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
    MASK = 'MASK'
    PHRASE_LENGTH_LIMIT = 5

    def __init__(self, relations_loader: RelationsLoader, vectorizer: Sent2VecVectorizer, sentence_map: dict):
        self.relations_loader = relations_loader
        self.vectorizer = vectorizer
        self.sentence_map = sentence_map

    def _is_phrase_too_long(self, member: Member) -> bool:
        return len(member.indices) > self.PHRASE_LENGTH_LIMIT

    @staticmethod
    def _make_key(label: str, id_domain: str, relation: Relation):
        id_document, member_from, member_to = relation
        return '\t'.join([
            label, id_domain, id_document,
            member_from.id_sentence, member_from.channel, str(member_from.indices), member_from.lemma,
            member_to.id_sentence, member_to.channel, str(member_to.indices), member_to.lemma,
            member_from.is_named_entity, member_to.is_named_entity
        ])

    def _mask_tokens(self, context: List[str], indices: Tuple[int]):
        return [self.MASK
                if index in indices else token
                for index, token in enumerate(context)]

    @staticmethod
    def _get_sentence_id_int(member: Member):
        return int(member.id_sentence.replace('sent', ''))

    def get_context_same_sentence(self, relation: Relation, document_sentences: dict) -> List[str]:
        id_document, member_from, member_to = relation

        sentence_from_index = self._get_sentence_id_int(member_from)
        sentence_to_index = self._get_sentence_id_int(member_to)

        left_sentence_index = sentence_from_index - 1
        context_left = document_sentences.get(left_sentence_index, [])

        to_map_indices = member_from.indices + member_to.indices
        context_between = document_sentences[sentence_from_index]
        context_between = self._mask_tokens(context_between, to_map_indices)

        right_sentence_index = sentence_to_index + 1
        context_right = document_sentences.get(right_sentence_index, [])

        return context_left + context_between + context_right

    def get_context_different_sentences(self, relation: Relation, document_sentences: dict) -> List[str]:
        id_document, member_from, member_to = relation

        sentence_from_index = self._get_sentence_id_int(member_from)
        sentence_to_index = self._get_sentence_id_int(member_to)

        if sentence_from_index < sentence_to_index:
            context = self.get_context_different_sentences_(
                member_from=member_from, member_to=member_to,
                document_sentences=document_sentences
            )
        else:  # sentence_to_index < sentence_from_index
            context = self.get_context_different_sentences_(
                member_from=member_to, member_to=member_from,
                document_sentences=document_sentences
            )

        return context

    def get_context_different_sentences_(self, member_from: Member, member_to: Member,
                                         document_sentences: dict) -> List[str]:
        sentence_from_index = self._get_sentence_id_int(member_from)
        sentence_to_index = self._get_sentence_id_int(member_to)

        entire_context_between = []

        left_sentence_index = sentence_from_index - 1
        context_left = document_sentences.get(left_sentence_index, [])

        context_between = document_sentences[sentence_from_index]
        context_between = self._mask_tokens(context_between, member_from.indices)
        entire_context_between.extend(context_between)

        for i in range(sentence_from_index + 1, sentence_to_index):
            context_between.extend(document_sentences.get(i, []))

        context_between = document_sentences[sentence_to_index]
        context_between = self._mask_tokens(context_between, member_to.indices)
        entire_context_between.extend(context_between)

        context_right = document_sentences.get(sentence_to_index + 1, [])

        return context_left + entire_context_between + context_right

    def make_map(self) -> [List, torch.tensor]:
        keys = []
        vectors = []

        for label, id_domain, relation in self.relations_loader.relations():
            id_document, member_from, member_to = relation

            if self._is_phrase_too_long(member_from) or self._is_phrase_too_long(member_to):
                continue

            document_sentences = self.sentence_map[(id_domain, id_document)]
            if member_from.id_sentence == member_to.id_sentence:
                context = self.get_context_same_sentence(relation, document_sentences)
            else:
                context = self.get_context_different_sentences(relation, document_sentences)

            key = self._make_key(label, id_domain, relation)
            vector = self.vectorizer.embed(context)

            keys.append(key)
            vectors.append(vector)

        return keys, torch.cat(vectors)


@click.command()
@click.option('--relations-file', required=True, type=str,
              help='Path to relations file.')
@click.option('--documents-files', required=True, type=str,
              help='Path to corpora relation files list.')
@click.option('--model', required=True, type=str,
              metavar='model.bin',
              help="Paths to sent2vec model.")
@click.option('--output-paths', required=True, type=(str, str),
              metavar='sent2vec.rel.keys sent2vec.rel.pt',
              help='Paths for saving keys and map files.')
def main(relations_file, documents_files, model, output_paths):
    vectorizer = Sent2VecVectorizer(model_path=model)
    relations_loader = RelationsLoader(relations_path=Path(relations_file))
    sentence_map = make_sentence_map(relations_paths=Path(documents_files))

    mapmaker = RelationsMapMaker(relations_loader, vectorizer, sentence_map)
    relations_keys, relations_vectors = mapmaker.make_map()

    keys_path, vectors_path = output_paths
    save_lines(Path(keys_path), relations_keys)
    save_tensor(Path(vectors_path), relations_vectors)


if __name__ == '__main__':
    main()
