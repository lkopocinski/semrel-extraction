import itertools
import json
import glob


PATH = '/home/lukaszkopocinski/Lukasz/SentiOne/korpusyneroweiaspektowe/*.json'
for file in glob.glob(PATH):
    print(f'\n\n\n{file}')
    with open(file) as json_file:
        data = json.load(json_file)
        for key, value in data.items():
            b = set()
            p = set()
            bi = set()
            pi = set()

            for item in value['ner']:
                s = item['source']
                if s['type'] == 'BRAND_NAME':
                    b.add(s['text'])
                elif s['type'] == 'PRODUCT_NAME':
                    p.add(s['text'])
                elif s['type'] == 'BRAND_NAME_IMP':
                    bi.add(s['text'])
                elif s['type'] == 'PRODUCT_NAME_IMP':
                    pi.add(s['text'])

                t = item['target']
                if t['type'] == 'BRAND_NAME':
                    b.add(t['text'])
                elif t['type'] == 'PRODUCT_NAME':
                    p.add(t['text'])
                elif t['type'] == 'BRAND_NAME_IMP':
                    bi.add(t['text'])
                elif t['type'] == 'PRODUCT_NAME_IMP':
                    pi.add(t['text'])

            for brand, product in itertools.product(b, p):
                if brand in product or product in brand:
                    print(f'\n{key}')
                    print(f'BRAND_NAME : {b}')
                    print(f'PRODUCT_NAME : {p}')
                    print(f'BRAND_NAME_IMP : {bi}')
                    print(f'PRODUCT_NAME_IMP : {pi}')
                    print(f'{brand} : {product}')



