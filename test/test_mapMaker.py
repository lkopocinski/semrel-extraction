from unittest import mock
from unittest.mock import PropertyMock

from scripts import MapMaker


@mock.patch('data.scripts.maps.Document', autospec=True)
@mock.patch('data.scripts.maps.DocSentence', autospec=True)
def test_make_keys(document, sentence):
    type(document).directory = PropertyMock(return_value='DIR')
    type(document).id = PropertyMock(return_value='DOC_ID')
    type(sentence).id = PropertyMock(return_value='SENT_ID')
    type(sentence).orths = PropertyMock(
        return_value=['orth0', 'orth1', 'orth2']
    )

    keys = MapMaker.make_keys(document, sentence)

    expected_keys = [
        ('DIR', 'DOC_ID', 'SENT_ID', 0),
        ('DIR', 'DOC_ID', 'SENT_ID', 1),
        ('DIR', 'DOC_ID', 'SENT_ID', 2)
    ]

    assert keys == expected_keys


@mock.patch('data.scripts.vectorizers.Vectorizer', autospec=True)
@mock.patch('data.scripts.maps.Document', autospec=True)
@mock.patch.object(
    MapMaker, 'make_sentence_map', autospec=True, return_value=[]
)
def test_make_document_map(vectorizer, document):
    type(document).sentences = PropertyMock(
        return_value=['sent1', 'sent2', 'sent3']
    )

    maker = MapMaker(vectorizer)
    maker.make_document_map(document)
