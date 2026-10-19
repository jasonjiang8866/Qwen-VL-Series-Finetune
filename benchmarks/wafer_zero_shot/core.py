"""CPU-only rendering, prompt construction, strict parsing and metrics."""
import math
import re
from PIL import Image, ImageDraw

INVALID = '__invalid__'


def validate_config(config):
    labels = config['labels']
    if not labels or any(not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', k) or not isinstance(v, str) or not v.strip() for k, v in labels.items()):
        raise ValueError('Labels require unique ASCII identifiers and nonempty definitions')
    for name, template in config['strategies'].items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or '{definitions}' not in template or '{labels}' not in template:
            raise ValueError('Strategies require safe names, {definitions} and {labels}')
    if not config['strategies']:
        raise ValueError('At least one strategy is required')
    return config


def render(coordinates, *, radius=1.0, center=(0.0, 0.0), size=448, margin=16, dot_radius=2):
    """Fixed physical frame, x right/y up. Reject invalid/out-of-wafer points."""
    if not math.isfinite(radius) or radius <= 0 or len(center) != 2 or not all(math.isfinite(v) for v in center):
        raise ValueError('Invalid wafer geometry')
    if size < 32 or margin < dot_radius + 2 or 2 * margin >= size or dot_radius < 1:
        raise ValueError('Invalid render dimensions')
    image = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(image)
    half = (size - 1) / 2
    scale = (half - margin) / radius
    draw.ellipse((margin, margin, size - 1 - margin, size - 1 - margin), outline='#808080', width=1)
    for point in coordinates:
        if len(point) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point):
            raise ValueError('Each coordinate must contain two finite numbers')
        x, y = point[0] - center[0], point[1] - center[1]
        if math.hypot(x, y) > radius * (1 + 1e-12):
            raise ValueError('Coordinate outside wafer; check center/radius/units')
        px, py = round(half + x * scale), round(half - y * scale)
        draw.ellipse((px-dot_radius, py-dot_radius, px+dot_radius, py+dot_radius), fill='black')
    return image


def messages(config, strategy):
    definitions = '\n'.join(f'{key}: {value}' for key, value in config['labels'].items())
    system = config['strategies'][strategy].format(definitions=definitions, labels=', '.join(config['labels']))
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': [
        {'type': 'image'}, {'type': 'text', 'text': 'Classify this inspection defect distribution map.'}]}]


def parse_label(text, labels):
    # Whitespace is the ONLY normalization. Never extract a label from prose.
    candidate = text.strip()
    return candidate if candidate in labels else None


def metrics(truth, predictions, labels):
    labels = list(labels)
    if not truth or len(truth) != len(predictions) or any(t not in labels for t in truth):
        raise ValueError('Nonempty aligned predictions and known ground truth required')
    matrix = [[0] * (len(labels) + 1) for _ in labels]
    for actual, predicted in zip(truth, predictions):
        matrix[labels.index(actual)][labels.index(predicted) if predicted in labels else len(labels)] += 1
    recalls, f1s, support = {}, [], {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        actual = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        recalls[label] = tp / actual if actual else 0.0
        support[label] = actual
        f1s.append(2 * tp / (actual + predicted) if actual + predicted else 0.0)
    return {'n': len(truth), 'accuracy': sum(matrix[i][i] for i in range(len(labels))) / len(truth),
            'macro_f1': sum(f1s) / len(labels), 'recall': recalls, 'support': support,
            'invalid_output_rate': sum(row[-1] for row in matrix) / len(truth),
            'confusion_matrix': {'rows_true': labels, 'columns_predicted': labels + [INVALID], 'counts': matrix}}
