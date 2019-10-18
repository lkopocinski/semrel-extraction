from .generate_negative_substitition import substitute_brand

BRAND = 'Narodowy Bank Polski'
CTX_BRAND = ['Alior Bank', 'ma', 'bardzo', 'dobre', 'warunki', 'na',
         'Konto', 'Jakze', 'Osobiste', '.']

CTX_PRODUCT = ['Alior', 'Bank', 'ma', 'bardzo', 'dobre', 'warunki', 'na',
         'Konto Jakze Osobiste', '.']
CTX_PRODUCT_2 = ['Zalozylem', 'Konto Jakze Osobiste', 'i', 'jestem', 'zadowolony', '.']

IDX_BRAND = 0
IDX_PRODUCT = 7
IDX_PRODUCT_2 = 1


def test_substitute_brand_when_diferrent_contexts():
    BRAND = 'Narodowy Bank Polski'
    CTX_BRAND = ['Alior Bank', 'ma', 'bardzo', 'dobre', 'warunki', 'na', 'Konto', 'Jakze', 'Osobiste', '.']
    CTX_PRODUCT = ['Zalozylem', 'Konto Jakze Osobiste', 'i', 'jestem', 'zadowolony', '.']

    IDX_BRAND = 0
    IDX_PRODUCT = 1

    idx_brand, ctx_brand, idx_product, ctx_product = substitute_brand(BRAND, IDX_BRAND, CTX_BRAND, IDX_PRODUCT, CTX_PRODUCT)

    assert 'Narodowy Bank Polski ma bardzo dobre warunki na Konto Jakze Osobiste .'.split(' ') == ctx_brand
    assert 'Zalozylem Konto Jakze Osobiste i jestem zadowolony .'.split(' ') == ctx_product

    assert idx_brand == IDX_BRAND
    assert idx_product == IDX_PRODUCT


def test_substitute_brand_when_same_contexts_one_word():
    BRAND = 'PKO'
    CTX_BRAND = ['Alior Bank', 'ma', 'bardzo', 'dobre', 'warunki', 'na', 'Konto', 'Jakze', 'Osobiste', '.']
    CTX_PRODUCT = ['Alior', 'Bank', 'ma', 'bardzo', 'dobre', 'warunki', 'na', 'Konto Jakze Osobiste', '.']

    IDX_BRAND = 0
    IDX_PRODUCT = 7

    idx_brand, ctx_brand, idx_product, ctx_product = substitute_brand(BRAND, IDX_BRAND, CTX_BRAND, IDX_PRODUCT, CTX_PRODUCT)

    assert 'PKO ma bardzo dobre warunki na Konto Jakze Osobiste .'.split(' ') == ctx_brand
    assert 'PKO ma bardzo dobre warunki na Konto Jakze Osobiste .'.split(' ') == ctx_product

    assert idx_brand == 0
    assert idx_product == 6


def test_substitute_brand_when_same_contexts_multiple_words_brand_first():
    BRAND = 'Narodowy Bank Polski'
    CTX_BRAND = ['Alior Bank', 'ma', 'bardzo', 'dobre', 'warunki', 'na', 'Konto', 'Jakze', 'Osobiste', '.']
    CTX_PRODUCT = ['Alior', 'Bank', 'ma', 'bardzo', 'dobre', 'warunki', 'na', 'Konto Jakze Osobiste', '.']

    IDX_BRAND = 0
    IDX_PRODUCT = 7

    idx_brand, ctx_brand, idx_product, ctx_product = substitute_brand(BRAND, IDX_BRAND, CTX_BRAND, IDX_PRODUCT, CTX_PRODUCT)

    assert 'Narodowy Bank Polski ma bardzo dobre warunki na Konto Jakze Osobiste .'.split(' ') == ctx_brand
    assert 'Narodowy Bank Polski ma bardzo dobre warunki na Konto Jakze Osobiste .'.split(' ') == ctx_product

    assert idx_brand == 0
    assert idx_product == 8
