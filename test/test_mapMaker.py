from unittest.mock import Mock, PropertyMock

from data.scripts.maps import MapMaker


def test_make_keys():
    vectorizer = Mock()

    document = Mock()
    type(document).directory = PropertyMock(return_value='DIR')
    type(document).id = PropertyMock(return_value='DOC_ID')

    sentence = Mock()
    type(sentence).id = PropertyMock(return_value='SENT_ID')
    type(sentence).orths = PropertyMock(return_value=['orth0', 'orth1', 'orth2', 'orth3'])

    maker = MapMaker(vectorizer)
    keys = maker.make_keys(document, sentence)

    expected_keys = [
        ('DIR', 'DOC_ID', 'SENT_ID', 0),
        ('DIR', 'DOC_ID', 'SENT_ID', 1),
        ('DIR', 'DOC_ID', 'SENT_ID', 2),
        ('DIR', 'DOC_ID', 'SENT_ID', 3)
    ]

    assert keys == expected_keys
