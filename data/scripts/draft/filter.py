from pathlib import Path


def lines(file_path: Path):
    with file_path.open('r', encoding='utf-8') as f:
        for line in f:
            yield line


def get_brands(file_path: Path):
    brands = set()
    for line in lines(file_path):
        row = line.strip().split('\t')

        if len(row) > 3:
            if row[4] == 'BRAND_NAME':
                brands.add(row[3])
            elif row[7] == 'BRAND_NAME':
                brands.add(row[6])
    return brands


def filter_(to_filter_path: Path, brands: set):
    with to_filter_path.open('r', encoding='utf-8') as f:
        for line in f:
            row = line.strip().split('\t')

            if len(row) > 3:
                if row[4] == 'BRAND_NAME' and row[3] not in brands:
                    print(line.strip())
                elif row[7] == 'BRAND_NAME' and row[6] not in brands:
                    print(line.strip())


if __name__ == '__main__':
    root = '../../../relextr/model/dataset/'
    train_file = Path(f'{root}/all/train.vectors')
    valid_file = Path(f'{root}/all/valid.vectors')
    test_file = Path(f'{root}/all/test.vectors')

    train_brands = get_brands(train_file)
    valid_brands = get_brands(valid_file)
    brands = train_brands.union(valid_brands)
    filter_(test_file, brands)
