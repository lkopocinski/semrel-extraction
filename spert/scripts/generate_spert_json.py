#!/usr/bin/env python3.6

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import click

from semrel.data.scripts import constant
from semrel.data.scripts.entities import Relation
from semrel.data.scripts.relations import RelationsLoader
from semrel.data.scripts.utils.io import save_json, load_json
from spert.scripts.entities import \
    SPERTDocument, \
    SPERTDocRelation, \
    SPERTRelation
from spert.scripts.mapper import \
    InSentenceSPERTMapper, \
    BetweenSentencesSPERTMapper
from spert.scripts.utils import in_same_context, make_relation_key, \
    split_relations


def map_relations(
        relations: List[Relation],
        in_sentence_mapper: InSentenceSPERTMapper,
        between_sentence_mapper: BetweenSentencesSPERTMapper
) -> List[dict]:
    same_context_relations = [
        relation
        for relation in relations
        if in_same_context(relation)
    ]

    diff_context_relations = [
        relation
        for relation in relations
        if not in_same_context(relation)
    ]

    spert_same_context_relations = [
        in_sentence_mapper.map(relation)
        for relation in same_context_relations
    ]

    spert_diff_context_relations = [
        between_sentence_mapper.map(relation)
        for relation in diff_context_relations
    ]

    spert_same_context_relations_keys = [
        make_relation_key(relation)
        for relation in same_context_relations
    ]

    spert_diff_context_relations_keys = [
        make_relation_key(relation)
        for relation in diff_context_relations
    ]

    spert_relations = (spert_same_context_relations
                       + spert_diff_context_relations)
    spert_keys = (spert_same_context_relations_keys
                  + spert_diff_context_relations_keys)

    return generate_spert_jsons(spert_relations, spert_keys)


def generate_spert_jsons(
        spert_relations: List[SPERTRelation],
        spert_keys: List[str]
) -> List[Dict]:
    documents = defaultdict(SPERTDocument)
    for relation, key in zip(spert_relations, spert_keys):
        document = documents[key]
        document.tokens = relation.tokens

        if relation.head not in document.entities:
            document.entities.append(relation.head)

        if relation.tail not in document.entities:
            document.entities.append(relation.tail)

        index_from = document.entities.index(relation.head)
        index_to = document.entities.index(relation.tail)

        document.relations.add(
            SPERTDocRelation(index_from, index_to, relation.relation_type)
        )

    return [document.to_dict() for document in documents.values()]


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
    relations = relations_loader.filter_relations(
        filter_label=constant.IN_RELATION_LABEL
    )

    in_sentence_mapper = InSentenceSPERTMapper()
    between_sentence_mapper = BetweenSentencesSPERTMapper()

    for run_id, run_indices in indices.items():
        run_relations = split_relations(run_indices, relations)

        for set_name, set_relations in run_relations.items():
            documents = map_relations(
                relations=set_relations,
                in_sentence_mapper=in_sentence_mapper,
                between_sentence_mapper=between_sentence_mapper
            )

            save_path = Path(f'{output_dir}/{run_id}/{set_name}.json')
            save_json(documents, save_path)


if __name__ == '__main__':
    main()
