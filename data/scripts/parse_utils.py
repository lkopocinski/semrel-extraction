class Relation:

    def __init__(self, line):
        self.line = line
        self._from = None
        self._to = None
        self._init_from_line()

    def _init_from_line(self):
        line = self.line.strip()
        by_tab = line.split('\t')

        lemma_from, lemma_to = by_tab[0].replace(' : ', ':').split(':', 1)
        channel_from, channel_to = by_tab[1].replace(' : ', ':').split(':', 1)
        index_from, context_from = by_tab[2].split(':', 1)
        index_to, context_to = by_tab[3].split(':', 1)

        context_from = eval(context_from)
        context_to = eval(context_to)

        index_from = int(index_from)
        index_to = int(index_to)

        self._from = self.Element(lemma_from, channel_from, index_from, context_from)
        self._to = self.Element(lemma_to, channel_to, index_to, context_to)

    def __str__(self):
        return f'{self._from.lemma} : {self._to.lemma}\t{self._from.channel} : {self._to.channel}\t{self._from.index}:{self._from.context}\t{self._to.index}:{self._to.context}'

    @property
    def source(self):
        return self._from

    @property
    def dest(self):
        return self._to

    class Element:
        def __init__(self, lemma, channel, index, context):
            self.lemma = lemma
            self.channel = channel
            self.index = index
            self.context = context

        def __str__(self):
            return f'{self.lemma}\t{self.channel}\t{self.index}:{self.context}'
