#!/usr/bin/env python3.6
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Dict

import click

from data.scripts.entities import Relation
from data.scripts.relations import RelationsLoader
from data.scripts.utils.io import save_json, load_json
from data.spert.entities import SPERTDocument, SPERTDocRelation
from data.spert.mapper import InSentenceSPERTMapper, BetweenSentencesSPERTMapper


def split_relations(indices: Dict, relations_dict: Dict) -> Dict:
    train_relations = [relation for index, relation in relations_dict.items()
                       if index in indices['train']]
    valid_relations = [relation for index, relation in relations_dict.items()
                       if index in indices['valid']]
    test_relations = [relation for index, relation in relations_dict.items()
                      if index in indices['test']]

    return {'train': train_relations,
            'valid': valid_relations,
            'test': test_relations}


def map_relations(relations: Iterator[Relation],
                  in_sentence_mapper: InSentenceSPERTMapper,
                  between_sentence_mapper: BetweenSentencesSPERTMapper):
    documents = defaultdict(SPERTDocument)

    for relation in relations:
        id_document, member_from, member_to = relation
        in_same_context = member_from.id_sentence == member_to.id_sentence

        id_from = relation.member_from.id_sentence
        id_to = relation.member_to.id_sentence
        key = f'{id_document}-{id_from}-{id_to}'

        if in_same_context:
            spert_relation = in_sentence_mapper.map(relation)
        else:
            spert_relation = between_sentence_mapper.map(relation)

        document = documents[key]
        document.tokens = spert_relation.tokens

        if spert_relation.head not in document.entities:
            document.entities.append(spert_relation.head)

        if spert_relation.tail not in document.entities:
            document.entities.append(spert_relation.tail)

        index_from = document.entities.index(spert_relation.head)
        index_to = document.entities.index(spert_relation.tail)

        document.relations.add(
            SPERTDocRelation(index_from, index_to, spert_relation.relation_type)
        )

    return documents


@click.command()
@click.option('--input-path', required=True, type=str,
              help='Path to relations file.')
@click.option('--indices-file', required=True, type=str,
              help='Path to indices file.')
@click.option('--output-dir', required=True, type=str,
              help='Paths for saving SPERT json file.')
def main(input_path, indices_file, output_dir):

    indices = load_json(Path(indices_file))
    relations_loader = RelationsLoader(Path(input_path))
    relations = relations_loader._filter_relations(filter_label='in_relation')

    in_sentence_mapper = InSentenceSPERTMapper()
    between_sentence_mapper = BetweenSentencesSPERTMapper()

    for run_id, run_indices in indices.items():
        print(f"\n\nRUN_ID: {run_id}")
        run_relations = split_relations(run_indices, relations)

        for set_name, set_relations in run_relations.items():
            print(f"SET_NAME: {set_name}", end=" ")
            documents = map_relations(
                relations=set_relations,
                in_sentence_mapper=in_sentence_mapper,
                between_sentence_mapper=between_sentence_mapper
            )

            documents = [document.to_dict() for document in documents.values()]

            save_json(documents, Path(f'{output_dir}/{run_id}/{set_name}.json'))


if __name__ == '__main__':
    main()
