PHRASE_LENGTH_LIMIT = 5

# corpus
SUBST_KEY = 'subst'
TAGSET_KEY = 'nkjp'
NAMED_ENTITY_KEY = 'NAMED_ENTITY'

# relations
IN_RELATION_LABEL = 'in_relation'
NO_RELATION_LABEL = 'no_relation'

CHANNELS = (('BRAND_NAME', 'PRODUCT_NAME'),
            ('PRODUCT_NAME', 'BRAND_NAME'))

HEADER = '\t'.join([
    'label',
    'id_domain',
    'id_document',

    'id_sentence_from',
    'lemma_from',
    'channel_from',
    'is_named_entity_from',
    'indices_from',
    'context_from',

    'id_sentence_to',
    'lemma_to',
    'channel_to',
    'is_named_entity_to',
    'indices_to',
    'context_to'
])

MASK_KEY = 'MASK'
