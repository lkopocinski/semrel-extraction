import argparse
import glob
import random

try:
    import argcomplete
except ImportError:
    argcomplete = None


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_dir', required=True,
                        help='A directory with corpora and relations files.')
    parser.add_argument('-t', '--target_dir', required=True,
                        help='A directory to save generated splits.')
    parser.add_argument('-p', '--prefix', required=True,
                        help='A prefix for saved file.')

    if argcomplete:
        argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def split(source_dir_path):
    path = f'{source_dir_path}/*.rel.xml'
    files = glob.glob(path)
    random.shuffle(files)
    return chunk(files)


def chunk(seq):
    avg = len(seq) / float(5)
    t_len = int(3 * avg)
    v_len = int(avg)
    return [seq[0:t_len], seq[t_len:t_len + v_len], seq[t_len + v_len:]]


def save_list(file_name, files_list):
    with open(file_name, 'w', encoding='utf-8') as f_out:
        for line in files_list:
            f_out.write(f'{line}\n')


def main(argv=None):
    args = get_args(argv)

    train, valid, test = split(args.source_dir)
    save_list(f'{args.target_dir}/train/{args.prefix}.list', train)
    save_list(f'{args.target_dir}/valid/{args.prefix}.list', valid)
    save_list(f'{args.target_dir}/test/{args.prefix}.list', test)


if __name__ == '__main__':
    main()
