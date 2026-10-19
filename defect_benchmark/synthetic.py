"""Seeded illustrative coordinate maps, NOT a model-quality benchmark dataset."""
import argparse
import json
import math
import random
from pathlib import Path
from PIL import Image, ImageDraw


def generate(output, seed=42, per_class=2):
    if per_class < 1:
        raise ValueError('per_class must be positive')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    labels = ['uniform', 'center_cluster', 'edge_ring', 'scratch', 'localized_cluster']
    rows = []
    for label in labels:
        for _ in range(per_class):
            points = []
            angle = rng.uniform(0, 2 * math.pi)
            for j in range(160):
                if label == 'uniform' or j < 12:
                    r, a = math.sqrt(rng.random()) * .97, rng.uniform(0, 2 * math.pi)
                    x, y = r * math.cos(a), r * math.sin(a)
                elif label == 'edge_ring':
                    r, a = rng.uniform(.82, .94), rng.uniform(0, 2 * math.pi)
                    x, y = r * math.cos(a), r * math.sin(a)
                elif label == 'scratch':
                    t, w = rng.uniform(-.83, .83), rng.gauss(0, .025)
                    x, y = t * math.cos(angle) - w * math.sin(angle), t * math.sin(angle) + w * math.cos(angle)
                else:
                    offset = .5 if label == 'localized_cluster' else 0
                    x, y = rng.gauss(offset * math.cos(angle), .10), rng.gauss(offset * math.sin(angle), .10)
                if x*x + y*y <= 1:
                    points.append([x, y])
            sid = f'sample_{len(rows):03d}'
            image = Image.new('RGB', (512, 512), 'white')
            draw = ImageDraw.Draw(image)
            draw.ellipse((16, 16, 496, 496), outline='#888888', width=2)
            for x, y in points:
                px, py = 256 + x * 240, 256 - y * 240
                draw.ellipse((px-1.5, py-1.5, px+1.5, py+1.5), fill='black')
            image.save(root / f'{sid}.png')
            (root / f'{sid}.coordinates.json').write_text(json.dumps(points))
            rows.append(dict(id=sid, image=f'{sid}.png', label=label))
    (root / 'manifest.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    (root / 'provenance.json').write_text(json.dumps(dict(synthetic=True, seed=seed, per_class=per_class,
        note='Procedural inspection coordinates; no production data; not representative of fab accuracy.'), indent=2))
    return root / 'manifest.jsonl'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--per-class', type=int, default=2)
    args = p.parse_args()
    generate(args.output, args.seed, args.per_class)
