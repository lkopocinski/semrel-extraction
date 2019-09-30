import glob
import random
import argparse

try:
    import argcomplete
except ImportError:
    argcomplete = None


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--source_directory', required=True,
                        help='A directory with corpora and relations files.')
    parser.add_argument('-o', '--output_directory', required=True,
                        help='A directory to save generated splits.')
    parser.add_argument('-p', '--prefix', required=True,
                        help='A prefix for saved file.')

    if argcomplete:
        argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def sets_split(root_path):
    path = f'{root_path}/*.rel.xml'
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

    train, valid, test = sets_split(args.source_directory)
    save_list(f'{args.output_directory}/train/list_{args.prefix}.txt', train)
    save_list(f'{args.output_directory}/valid/list_{args.prefix}.txt', valid)
    save_list(f'{args.output_directory}/test/list_{args.prefix}.txt', test)


if __name__ == '__main__':
    main()
