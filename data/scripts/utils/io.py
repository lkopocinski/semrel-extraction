def save_lines(file_path, lines):
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open('w', encoding='utf-8') as f:
        for line in lines:
            f.write(f'{line}\n')
