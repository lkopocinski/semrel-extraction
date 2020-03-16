PHRASE_LENGTH_LIMIT = 5

# corpus
SUBST_KEY = 'subst'
TAGSET_KEY = 'nkjp'
NAMED_ENTITY_KEY = 'NAMED_ENTITY'

MASK_KEY = 'MASK'

# relations
IN_RELATION_LABEL = 'in_relation'
NO_RELATION_LABEL = 'no_relation'

CHANNELS = (('BRAND_NAME', 'PRODUCT_NAME'),
            ('PRODUCT_NAME', 'BRAND_NAME'))


# header
class RelationHeader:
    LABEL = 'label'
    DOMAIN = 'id_domain'
    DOCUMENT = 'id_document'

    # member from
    SENTENCE_FROM = 'id_sentence_from',
    LEMMA_FROM = 'lemma_from',
    CHANNEL_FROM = 'channel_from',
    NAMED_ENTITY_FROM = 'is_named_entity_from',
    INDICES_FROM = 'indices_from',
    CONTEXT_FROM = 'context_from',

    # member to
    SENTENCE_TO = 'id_sentence_to',
    LEMMA_TO = 'lemma_to',
    CHANNEL_TO = 'channel_to',
    NAMED_ENTITY_TO = 'is_named_entity_to',
    INDICES_TO = 'indices_to',
    CONTEXT_TO = 'context_to'

    HEADER = '\t'.join([
        LABEL,
        DOMAIN,
        DOCUMENT,

        # member from
        SENTENCE_FROM,
        LEMMA_FROM,
        CHANNEL_FROM,
        NAMED_ENTITY_FROM,
        INDICES_FROM,
        CONTEXT_FROM,

        # member to
        SENTENCE_TO,
        LEMMA_TO,
        CHANNEL_TO,
        NAMED_ENTITY_TO,
        INDICES_TO,
        CONTEXT_TO,
    ])
