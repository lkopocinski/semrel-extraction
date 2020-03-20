class Predictor(object):

    def __init__(self, net_model, elmo, fasttext, device='cpu'):
        self._net = net_model
        self._elmo = elmo
        self._fasttext = fasttext
        self.device = device

    def _make_vectors(self, pair):
        (idx1, ctx1), (idx2, ctx2) = pair
        print(pair)
        ev1 = self._elmo.embed(ctx1)[idx1]
        ev2 = self._elmo.embed(ctx2)[idx2]

        fv1 = self._fasttext.embed(ctx1)[idx1]
        fv2 = self._fasttext.embed(ctx2)[idx2]

        v = torch.cat([ev1, ev2, fv1, fv2])
        return v.to(self.device)

    def _predict(self, vectors):
        with torch.no_grad():
            prediction = self._net(vectors)
            prediction = torch.argmax(prediction)
            return prediction.item()

    def predict(self, pair):
        vectors = self._make_vectors(pair)
        return self._predict(vectors)