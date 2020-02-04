from data.scripts.generator import generate_negative2
from pathlib import Path


def test_generate_negative():
    paths = [Path('./test.rel.xml')]
    expected = []

    actual = generate_negative2(paths)

