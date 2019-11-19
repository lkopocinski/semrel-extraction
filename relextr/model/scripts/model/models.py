import numpy as np


class RelationVec:

    def __init__(self, line):
        self.line = line
        self._from = None
        self._to = None
        self.label = None
        self._init_from_line()

    def _init_from_line(self):
        line = self.line.strip()
        by_tab = line.split('\t')

        self.label = by_tab[0]
        vector_from, vector_to = np.array(eval(by_tab[1])), np.array(eval(by_tab[2]))
        lemma_from, lemma_to = by_tab[3], by_tab[6]
        channel_from, channel_to = by_tab[4], by_tab[7]
        index_from, context_from = by_tab[5].split(':', 1)
        index_to, context_to = by_tab[8].split(':', 1)
        conv_vector_from, conv_vector_to = np.array(eval(by_tab[9])), np.array(eval(by_tab[10]))
        ne_from, ne_to = eval(by_tab[11]), eval(by_tab[12])

        context_from = eval(context_from)
        context_to = eval(context_to)

        index_from = int(index_from)
        index_to = int(index_to)

        self._from = self.Element(vector_from, lemma_from, channel_from, index_from, context_from, conv_vector_from, ne_from)
        self._to = self.Element(vector_to, lemma_to, channel_to, index_to, context_to, conv_vector_to, ne_to)

    @property
    def source(self):
        return self._from

    @property
    def dest(self):
        return self._to

    class Element:
        def __init__(self, vector, lemma, channel, index, context, conv_vector, ne):
            self.vector = vector
            self.lemma = lemma
            self.channel = channel
            self.index = index
            self.context = context
            self.conv_vector = conv_vector
            self.ne = ne
