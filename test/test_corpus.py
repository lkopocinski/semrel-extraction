from pathlib import Path
from unittest.mock import MagicMock
from data.scripts.utils.corpus import Document, relations_documents_gen


def test_documents_gen():
    document = MagicMock()
    document.__eq__.return_value = 'test'

    relations_files = [Path('./test.rel.xml')]
    expected_documents = [document]

    actual_documents = [relations_documents_gen(relations_files)]
    assert actual_documents == expected_documents
