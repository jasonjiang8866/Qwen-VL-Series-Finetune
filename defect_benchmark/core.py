"""CPU-only validation, prompts, constraints and metrics."""
import hashlib
import json
import re
from .render import render
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    classes = cfg.get('classes', [])
    ids = [c['id'] for c in classes]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('classes must have unique, nonempty IDs')
    for c in classes:
        if not re.fullmatch(r'[a-z][a-z0-9_]*', c['id']) or not c.get('description', '').strip():
            raise ValueError('class IDs must be lowercase identifiers with descriptions')
    if not isinstance(cfg.get('rules'), list) or not all(isinstance(x, str) for x in cfg['rules']):
        raise ValueError('rules must be a list of strings')
    if not isinstance(cfg.get('system_prompt'), str) or not cfg['system_prompt'].strip():
        raise ValueError('system_prompt required')
    return cfg


STRATEGIES = {
    "definitions": "Use the class definitions and decision rules below.",
    "geometry": "Compare central concentration, radial distribution, linear alignment and spatial spread internally.",
    "contrastive": "Compare the best matching class against alternatives; reject classes whose defining geometry is absent.",
}


def build_prompt(cfg, strategy="definitions"):
    if strategy not in STRATEGIES:
        raise ValueError("unknown prompt strategy")
    return '\n'.join([
        cfg['system_prompt'], STRATEGIES[strategy],
        'This is a Defect Distribution Map: dots are inspection defect coordinates.',
        'It is NOT a Wafer Bin Map. Do not infer die pass/fail, yield or physical root cause.',
        'Classes (choose exactly one ID):',
        *[f"{c['id']}: {c['description']}" for c in cfg['classes']],
        'Decision rules:', *cfg['rules'],
        'Return only the exact class ID. No JSON, explanation, punctuation or markdown.',
    ])


def load_manifest(path, classes, rendered_dir=None, render_options=None):
    """One manifest contract: exactly one of image or coordinates per row.

    Coordinates are inline x/y pairs with explicit center/radius in the same units.
    Optional rendered_dir freezes all model inputs, including image-only rows.
    """
    from PIL import Image
    path = Path(path).resolve()
    rows, seen = [], set()
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get('id'), str) or not row['id'] or row['id'] in seen:
            raise ValueError(f'line {lineno}: missing/duplicate ID')
        if row.get('label') not in classes:
            raise ValueError(f'line {lineno}: unknown ground truth')
        if 'synthetic' in row and type(row['synthetic']) is not bool:
            raise ValueError('synthetic must be a boolean when supplied')
        if ('image' in row) == ('coordinates' in row):
            raise ValueError('provide exactly one of image or coordinates')
        record = dict(id=row['id'], label=row['label'], synthetic=row.get('synthetic'))
        destination = None
        if rendered_dir is not None:
            Path(rendered_dir).mkdir(parents=True, exist_ok=True)
            destination = Path(rendered_dir) / f'{len(rows):06d}.png'
        if 'coordinates' in row:
            if destination is None:
                raise ValueError('coordinate input requires rendered_dir')
            image = render(row['coordinates'], radius=row['radius'], center=row['center'],
                           **(render_options or {}))
            image.save(destination)
            record.update(input_type='coordinates', coordinates=row['coordinates'],
                          center=row['center'], radius=row['radius'])
        else:
            source = Path(row['image'])
            if not source.is_absolute():
                source = path.parent / source
            source = source.resolve()
            with Image.open(source) as im:
                im.verify()
            record.update(input_type='image', source_image_sha256=sha256(source))
            if destination is not None:
                # Canonical RGB pixels; ignore orientation metadata consistently.
                with Image.open(source) as im:
                    im.convert('RGB').save(destination)
            else:
                destination = source
        record.update(image=str(destination.resolve()), image_sha256=sha256(destination))
        rows.append(record)
        seen.add(row['id'])
    if not rows:
        raise ValueError('empty dataset')
    return rows


def parse_output(raw, classes):
    value = raw.strip()
    return value if value in classes else None


class LabelConstraint:
    """Token-prefix restriction, including labels that prefix other labels."""
    def __init__(self, tokenizer, labels, prompt_length, eos_id):
        self.start, self.eos = prompt_length, eos_id
        self.sequences = [tokenizer.encode(x, add_special_tokens=False) for x in labels]
        if eos_id is None or any(not x or eos_id in x for x in self.sequences):
            raise ValueError('invalid label tokenization or missing EOS')
        if len({tuple(x) for x in self.sequences}) != len(labels):
            raise ValueError('label tokenization collision')
        for label, ids in zip(labels, self.sequences):
            if tokenizer.decode(ids, skip_special_tokens=True).strip() != label:
                raise ValueError('labels must round-trip through tokenizer')
        self.max_tokens = max(map(len, self.sequences)) + 1

    def allowed(self, prefix):
        allowed = set()
        for seq in self.sequences:
            if seq[:len(prefix)] == prefix:
                allowed.add(seq[len(prefix)] if len(prefix) < len(seq) else self.eos)
        if not allowed:
            raise ValueError('generation left label prefix tree')
        return sorted(allowed)

    def __call__(self, batch_id, input_ids):
        return self.allowed(input_ids[self.start:].tolist())


def metrics(truth, predictions, classes):
    if not truth or len(truth) != len(predictions):
        raise ValueError('nonempty aligned truth/predictions required')
    matrix = [[0] * (len(classes) + 1) for _ in classes]
    for y, p in zip(truth, predictions):
        matrix[classes.index(y)][classes.index(p) if p in classes else len(classes)] += 1
    per_class = {}
    for i, label in enumerate(classes):
        tp = matrix[i][i]
        fn = sum(matrix[i]) - tp
        fp = sum(row[i] for row in matrix) - tp
        per_class[label] = {'support': sum(matrix[i]),
                            'recall': tp / (tp + fn) if tp + fn else 0.0,
                            'precision': tp / (tp + fp) if tp + fp else 0.0, 'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
    return {
        'n': len(truth), 'accuracy': sum(matrix[i][i] for i in range(len(classes))) / len(truth),
        'macro_f1': sum(x['f1'] for x in per_class.values()) / len(classes),
        'invalid_output_rate': sum(row[-1] for row in matrix) / len(truth),
        'per_class': per_class,
        'confusion_matrix': {'rows_true': classes, 'columns_predicted': classes + ['__invalid__'], 'counts': matrix},
    }
