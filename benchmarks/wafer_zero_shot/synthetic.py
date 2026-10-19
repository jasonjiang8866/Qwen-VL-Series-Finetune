"""Seeded toy geometry for pipeline validation, not fab performance."""
import math
import random


def generate(config, per_class=4, seed=42):
    if per_class < 1: raise ValueError('per_class must be positive')
    rng = random.Random(seed)
    records = []
    for label in config['labels']:
        pattern = config['synthetic_patterns'][label]
        if pattern not in ('center', 'ring', 'scratch', 'random', 'empty'):
            raise ValueError(f'Unknown synthetic pattern: {pattern}')
        for i in range(per_class):
            points = []
            for _ in range(0 if pattern == 'empty' else 80):
                if pattern == 'scratch':
                    x, y = rng.uniform(-.8, .8), rng.gauss(0, .015)
                else:
                    angle = rng.uniform(0, 2 * math.pi)
                    r = (.25 * math.sqrt(rng.random()) if pattern == 'center' else
                         rng.uniform(.72, .82) if pattern == 'ring' else .95 * math.sqrt(rng.random()))
                    x, y = r * math.cos(angle), r * math.sin(angle)
                points.append([x, y])
            records.append({'id': f'synthetic-{label}-{i:04d}', 'label': label,
                            'coordinates': points, 'radius': 1.0, 'center': [0, 0], 'synthetic': True})
    rng.shuffle(records)
    return records
