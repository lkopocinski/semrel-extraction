def is_better_fscore(fscore, best_fscore):
    return fscore[0] > best_fscore[0] and fscore[1] > best_fscore[1]


def labels2idx(labels):
    mapping = {
        'no_relation': 0,
        'in_relation': 1,
    }
    return [mapping[label] for label in labels if label in mapping]
