def save_lines(path, lines):
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open('w', encoding='utf-8') as f:
        for line in lines:
            f.write(f'{line}\n')


def load_file(path):
    with path.open('r', encoding='utf-8') as f:
        return [line.strip() for line in f]
