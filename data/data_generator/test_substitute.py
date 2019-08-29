import pytest

from .gen_negative_substituted import substitute_brand


def test_substitute_brand():
    brand = 'Full brand name'
    brand_idx = 2
    brand_ctx = ['This', 'is', 'Full brand name', 'and', 'it', 'is', 'awesome']

    product = 'Excellent product'
    product_idx = 3
    product_ctx = ['I', 'love', 'an', 'Excellent product', 'very', 'much']

    actual_idx_brand, actual_ctx_brand, actual_idx_product, actual_ctx_product = substitute_brand(brand, brand_idx, brand_ctx, product_idx, product_ctx)

    expected_idx_brand = brand_idx
    expected_ctx_brand = ['This', 'is', 'Full', 'brand', 'name', 'and', 'it', 'is', 'awesome']
    expected_idx_product = product_idx
    expected_ctx_product = ['I', 'love', 'an', 'Excellent', 'product', 'very', 'much']

    assert actual_idx_brand == expected_idx_brand
    assert actual_ctx_brand == expected_ctx_brand
    assert actual_idx_product == expected_idx_product
    assert actual_ctx_product == expected_ctx_product
