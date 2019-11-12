import math
import random
from collections import defaultdict

from utils.io import load_file
from model.models import Relation


def sample_positive(file_path, batch_size):
    lines = load_file(file_path)
    if len(lines) > batch_size:
        lines = random.sample(lines, batch_size)
    yield lines


def sample_negative(file_path, batch_size):
    size = math.floor(batch_size / 3)
    lines = load_file(file_path)

    type_dict = defaultdict(list)
    for line in lines:
        relation = Relation.from_line(line)
        if relation.source.channel == '' and relation.dest.channel == '':
            type_dict['plain'].append(f'{relation}')
        elif relation.source.channel == 'BRAND_NAME' and relation.dest.channel == '':
            type_dict['brand'].append(f'{relation}')
        elif relation.source.channel == '' and relation.dest.channel == 'PRODUCT_NAME':
            type_dict['product'].append(f'{relation}')

    out_lines = []
    for key, lines in type_dict.items():
        if len(lines) > size:
            lines = random.sample(lines, size)
        out_lines.extend(lines)

    yield out_lines
