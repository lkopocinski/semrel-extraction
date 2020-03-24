from itertools import chain
from pathlib import Path
from typing import Iterator, List

from corpus_ccl import cclutils as ccl

from semrel.data.scripts.corpus import Document


def from_index_documents_gen(relations_files_index: Path) -> Iterator[Document]:
    with relations_files_index.open('r', encoding='utf-8') as file:
        relations_files = [Path(line.strip()) for line in file]
        return documents_gen(relations_files)


def documents_gen(relations_files: Iterator[Path]) -> Iterator[Document]:
    for rel_path in relations_files:
        ccl_name = rel_path.name.replace('.rel', '.ner')
        ccl_path = rel_path.parent / ccl_name
        if rel_path.is_file() and ccl_path.is_file():
            ccl_document = ccl.read_ccl_and_rel_ccl(
                ccl_file=str(ccl_path), rel_ccl_file=str(rel_path)
            )
            yield Document(ccl_document)


def relations_files_paths(
        corpora_path: str, directories: List
) -> Iterator[Path]:
    return list(chain.from_iterable(
        dir_path.glob('*.rel.xml')
        for dir_path in Path(corpora_path).iterdir()
        if dir_path.stem in directories))
