import torch

from data.scripts.maps import MapLoader

KEYS_FILE = './test.map.keys'
VECTORS_FILE = './test.map.pt'


def test_load_map():
    expected_keys = {
        ('116', '00198000', 'sent1', 0): 0,
        ('116', '00198000', 'sent1', 1): 1,
        ('116', '00198000', 'sent1', 2): 2,
        ('116', '00198000', 'sent1', 3): 3
    }
    expected_vectors = torch.FloatTensor([
        [0.1070, -0.1315, -0.1719, -0.1000, 0.0375],
        [0.1972, -0.1865, -0.1212, 0.0129, -0.0543],
        [0.2795, 0.0798, -0.0098, -0.1617, 0.0672],
        [0.2459, 0.0555, -0.0038, -0.3139, -0.3428]
    ])

    map_loader = MapLoader(
        keys_file=KEYS_FILE,
        vectors_file=VECTORS_FILE
    )

    keys, vectors = map_loader()

    assert keys == expected_keys
    assert repr(expected_vectors) == repr(vectors)
