import numpy as np

def load_batches(datapath, batch_size=15):
    with open(datapath, encoding="utf-8") as ifile:
        dataset = []
        batch = []
        for ind, line in enumerate(ifile, 1):
            row = line.strip().split('\t')
            cls = row[0]
            v1, v2 = np.array(eval(row[1])), np.array(eval(row[2]))
            if (ind % batch_size) == 0:
                dataset.append(batch)
                batch = []
            vdiff = v1 - v2
            batch.append((cls, np.concatenate([v1, v2, vdiff])))
        if batch:
            dataset.append(batch)
        return dataset
