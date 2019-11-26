class RelationVec:

    def __init__(self, line):
        self.line = line
        self._from = None
        self._to = None
        self.label = None
        self._init_from_line()

    def _init_from_line(self):
        row = self.line.strip().split('\t')

        self.label = row[0]
        self.lemma1, self.lemma2 = row[1], row[2]
        self.channel1, self.channel2 = row[3], row[4]
        self.ne1, self.ne2 = row[5], row[6]
        self.indices1, self.indices2 = eval(row[7]), eval(row[8])
        self.context1, self.context2 = eval(row[9]), eval(row[10])

        self.elmo1, self.elmo2 = eval(row[11]), row[12]
        self.elmoconv1, self.elmoconv2 = eval(row[13]), eval(row[14])
        self.fasttext1, self.fasttext2 = eval(row[15]), eval(row[16])
        self.sent2vec1, self.sent2vec2 = eval(row[17]), eval(row[18])
        self.retrofit1, self.retrofit2 = eval(row[19]), eval(row[20])
        self.ner1, self.ner2 = eval(row[21]), eval(row[22])
