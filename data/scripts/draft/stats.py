import json
from collections import defaultdict
from pathlib import Path

PATH = Path('/home/lukaszkopocinski/Lukasz/SentiOne/korpusyneroweiaspektowe/')


def corpus_files(path: Path):
    for file in path.glob('ner_*_export.json'):
        print(f'\n --- {file.stem} ---')
        with file.open('r', encoding='utf-8') as json_file:
            yield json.load(json_file)


def main():
    for data in corpus_files(PATH):
        brand_dict = defaultdict(int)
        for id, content in data.items():
            id = content['id']
            text = content['text']
            relations = content['ner']

            for rel in relations:
                s = rel['source']
                if s['type'] == 'BRAND_NAME':
                    brand_dict[s['text']] += 1

                t = rel['target']
                if t['type'] == 'BRAND_NAME':
                    brand_dict[t['text']] += 1
        brands = set(brand_dict.keys())
        print(brands)


if __name__ == '__main__':
    main()
