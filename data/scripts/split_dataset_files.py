import glob
import random

ROOT_PATH = '/home/lukaszkopocinski/Lukasz/SentiOne/korpusyneroweiaspektowe'
FILES_NR = (81, 82, 83)
SAVE_DIR = '../splits'


def chunk(seq):
    avg = len(seq) / float(5)
    t_len = int(3 * avg)
    v_len = int(avg)
    return [seq[0:t_len], seq[t_len:t_len + v_len], seq[t_len + v_len:]]


def sets_split(root_path, nr):
    path = f'{root_path}/inforex_export_{nr}/documents/*.rel.xml'
    files = glob.glob(path)
    random.shuffle(files)
    return chunk(files)


def save_list(file_name, files_list):
    with open(file_name, 'w', encoding='utf-8') as f_out:
        for line in files_list:
            f_out.write(f'{line}\n')


if __name__ == '__main__':
    for nr in FILES_NR:
        train, valid, test = sets_split(ROOT_PATH, nr)
        save_list(f'{SAVE_DIR}/train_files_{nr}.txt', train)
        save_list(f'{SAVE_DIR}/valid_files_{nr}.txt', valid)
        save_list(f'{SAVE_DIR}/test_files_{nr}.txt', test)
