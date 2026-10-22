"""Reproducible synthetic morphology stress suite; never a fab accuracy dataset."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

from .core import load_config, metrics, sha256
from .render import render
from .synthetic import generate as generate_smoke

CONFIG = Path(__file__).parent / 'configs/example.json'
SCHEMA = 'ddm-morphology-stress-v1'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def derived_seed(seed, key):
    """Independent streams, unaffected by changes to other perturbation families."""
    return int(fingerprint([SCHEMA, seed, key])[:16], 16)


def disk_points(rng, count):
    points = []
    for _ in range(count):
        r, a = math.sqrt(rng.random()) * .97, rng.uniform(0, 2 * math.pi)
        points.append([r * math.cos(a), r * math.sin(a)])
    return points


def generate(output, seed=42, replicates=3):
    if type(seed) is not int or type(replicates) is not int or replicates < 1:
        raise ValueError('integer seed and positive integer replicates required')
    cfg = load_config(CONFIG)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    render_options = cfg['render']

    def emit(points, label, source_class, reference_id, family, level, params,
             sample_seed, source_seed, replicate, rationale):
        sid = f'sample_{len(rows):05d}'
        stress = dict(schema=SCHEMA, suite_seed=seed, seed=sample_seed, source_seed=source_seed,
                      replicate=replicate, reference_id=reference_id, source_class=source_class,
                      role='anchor' if family == 'baseline' else 'variant', family=family,
                      level=level, parameters=params, point_count=len(points),
                      coordinates_sha256=fingerprint(points), config_sha256=fingerprint(cfg),
                      semantic_preserving=reference_id is not None and label is not None,
                      evaluation='exploratory' if label is None else 'labeled', rationale=rationale)
        render(points, **render_options).save(root / f'{sid}.png')
        rows.append(dict(id=sid, coordinates=points, center=[0.0, 0.0], radius=1.0,
                         label=label, synthetic=True, stress=stress))
        return sid

    for replicate in range(replicates):
        source_seed = derived_seed(seed, f'replicate/{replicate}')
        # Reuse the original smoke generator unchanged, including its 12 background dots.
        base = root / 'anchors' / f'{replicate:03d}'
        generate_smoke(base, seed=source_seed, per_class=1)
        sources = [json.loads(line) for line in (base / 'coordinates.jsonl').read_text().splitlines()]
        by_class = {r['label']: r['coordinates'] for r in sources}
        anchors = {}
        for row in sources:
            label, points = row['label'], row['coordinates']
            n = len(points)
            anchor = emit(points, label, label, None, 'baseline', 'original',
                          dict(requested_count=160), source_seed, source_seed, replicate,
                          'Original illustrative single-pattern synthetic class; not a fab annotation.')
            anchors[label] = anchor

            def variant(family, level, transform, preserving, rationale):
                key = f'{replicate}/{label}/{family}/{level}'
                perturb_seed = derived_seed(seed, key)
                changed, params = transform(random.Random(perturb_seed))
                params = dict(params, source_count=n)
                # Clipping can change visible morphology: never score it as preserving.
                keep_label = preserving and params.get('dropped_points', 0) == 0
                return emit(changed, label if keep_label else None, label, anchor,
                            family, level, params, perturb_seed, source_seed, replicate,
                            rationale if keep_label or not preserving else
                            'Translation clipped points; retained as exploratory only.')

            variant('density', 'half', lambda rng: (rng.sample(points, n // 2),
                    dict(method='subsample_without_replacement', target_count=n // 2)), True,
                    'Moderate thinning is assumed to preserve this strong illustrative morphology.')

            def denser(rng):
                extra = []
                for x, y in points:
                    # Bounded local jitter adds points without exact duplicates or frame changes.
                    while True:
                        u, v = x + rng.uniform(-.005, .005), y + rng.uniform(-.005, .005)
                        if u*u + v*v <= 1:
                            extra.append([u, v])
                            break
                return points + extra, dict(method='local_jitter_densification',
                                            jitter_half_width=.005, target_count=2*n)
            variant('density', 'double', denser, True,
                    'Local densification preserves arrangement; not independent new defects.')
            for fraction in (.1, 1.0):
                added = round(n * fraction)
                variant('noise', f'add_{fraction:g}', lambda rng, added=added: (
                        points + disk_points(rng, added), dict(added_count=added,
                        added_to_source_ratio=added/n, distribution='uniform_disk_rmax_0.97')),
                        fraction == .1,
                        'Sparse added background is ignored by the example rules.' if fraction == .1 else
                        'Heavy background may remove dominance; no assigned truth.')
            for degrees in (30, 90, 180):
                a = math.radians(degrees)
                rotated = [[x*math.cos(a)-y*math.sin(a), x*math.sin(a)+y*math.cos(a)] for x, y in points]
                variant('rotation', f'{degrees}_deg', lambda rng, rotated=rotated, degrees=degrees:
                        (rotated, dict(degrees=degrees, pivot=[0, 0])), True,
                        'Rigid rotation about wafer center preserves the example class definitions.')
            for dx in (.05, .4):
                def shifted(rng, dx=dx):
                    moved = [[x+dx, y] for x, y in points]
                    inside = [p for p in moved if math.hypot(*p) <= 1]
                    return inside, dict(dx=dx, dy=0, units='wafer_radius',
                                        dropped_points=n-len(inside), clipping='discard_outside_disk')
                variant('translation', f'x_{dx:g}', shifted,
                        dx == .05 and label in ('center_cluster', 'localized_cluster', 'scratch'),
                        'Small unclipped cluster/line displacement assumed to preserve morphology.' if dx == .05 else
                        'Large displacement can change center/edge semantics; no assigned truth.')
            variant('sparse', 'four_points', lambda rng: (rng.sample(points, 4),
                    dict(target_count=4, method='subsample_without_replacement')), False,
                    'Too few defects to impose an unambiguous example class.')

        # Ring completeness is undefined; no arbitrary priority-rule ground truth.
        for degrees in (270, 180, 90):
            ring_seed = derived_seed(seed, f'{replicate}/partial_ring/{degrees}')
            rng = random.Random(ring_seed)
            start = rng.uniform(0, 2*math.pi)
            points = []
            for _ in range(148):
                r, a = rng.uniform(.82, .94), start + rng.uniform(0, math.radians(degrees))
                points.append([r*math.cos(a), r*math.sin(a)])
            points += disk_points(rng, 12)
            emit(points, None, 'edge_ring', anchors['edge_ring'], 'partial_ring', f'{degrees}_deg',
                 dict(arc_degrees=degrees, start_radians=start, radial_range=[.82, .94],
                      structured_count=148, background_count=12, method='resample_arc'),
                 ring_seed, source_seed, replicate,
                 'Ring completeness threshold is not defined; record prediction without accuracy.')

        for first, second in (('scratch', 'edge_ring'), ('center_cluster', 'localized_cluster'),
                              ('edge_ring', 'localized_cluster')):
            for fraction in (.25, .5, .75):
                mixed_seed = derived_seed(seed, f'{replicate}/mixed/{first}/{second}/{fraction}')
                rng = random.Random(mixed_seed)
                count = round(160 * fraction)
                points = rng.sample(by_class[first], count) + rng.sample(by_class[second], 160-count)
                rng.shuffle(points)
                emit(points, None, first, anchors[first], 'mixed', f'{first}+{second}:{fraction:g}',
                     dict(components=[first, second], counts=[count, 160-count], total_count=160,
                          first_fraction=fraction, component_source_seed=source_seed),
                     mixed_seed, source_seed, replicate,
                     'Competing morphologies: prompt priority may affect prediction, but supplies no truth.')
        emit([], None, 'uniform', anchors['uniform'], 'blank', 'empty', dict(target_count=0),
             derived_seed(seed, f'{replicate}/blank'), source_seed, replicate,
             'The illustrative taxonomy has no blank class.')

    coordinates = root / 'coordinates.jsonl'
    coordinates.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    images = [dict(id=r['id'], image=f"{r['id']}.png", label=r['label'], synthetic=True,
                   stress=r['stress']) for r in rows]
    (root / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in images))
    validate_suite(rows, cfg)
    (root / 'provenance.json').write_text(json.dumps(dict(schema=SCHEMA, seed=seed,
        replicates=replicates, samples=len(rows), config=cfg, render=render_options,
        config_sha256=fingerprint(cfg), coordinates_sha256=sha256(coordinates),
        manifest_sha256=sha256(root / 'manifest.jsonl'),
        python_rng='random.Random with SHA256-derived per-transformation seeds',
        semantic_policy='Illustrative assumptions only; inspect rendered examples before interpreting scores.',
        real_fab_accuracy_validated=False, note='Synthetic coordinates only; no production data.'), indent=2))
    return root / 'manifest.jsonl'


def validate_suite(rows, cfg):
    """Fail before backend loading on broken pairs, unlabeled leakage or config drift."""
    lookup = {r['id']: r for r in rows}
    if not rows or len(lookup) != len(rows):
        raise ValueError('stress suite requires unique sample IDs')
    suite_seeds = set()
    for row in rows:
        s = row.get('stress')
        if not isinstance(s, dict) or s.get('schema') != SCHEMA or row.get('synthetic') is not True:
            raise ValueError('explicit synthetic stress schema required')
        if s.get('config_sha256') != fingerprint(cfg):
            raise ValueError('stress config differs from generator taxonomy/render settings')
        for key in ('seed', 'source_seed', 'suite_seed', 'replicate', 'point_count'):
            if type(s.get(key)) is not int or (key != 'suite_seed' and s[key] < 0):
                raise ValueError(f'invalid stress {key}')
        suite_seeds.add(s['suite_seed'])
        if not isinstance(s.get('parameters'), dict) or not s.get('rationale'):
            raise ValueError('stress parameters and semantic rationale required')
        if not all(isinstance(s.get(k), str) and s[k] for k in ('family', 'level', 'source_class')):
            raise ValueError('stress family, level and source class required')
        if s['source_class'] not in [c['id'] for c in cfg['classes']]:
            raise ValueError('unknown stress source class')
        if 'coordinates' in row and (s['point_count'] != len(row['coordinates']) or
                                    s.get('coordinates_sha256') != fingerprint(row['coordinates'])):
            raise ValueError('stress coordinates do not match recorded count/hash')
        if type(s.get('semantic_preserving')) is not bool:
            raise ValueError('semantic_preserving must be boolean')
        if s.get('evaluation') != ('exploratory' if row['label'] is None else 'labeled'):
            raise ValueError('evaluation scope must agree with nullable label')
        if s.get('role') == 'anchor':
            if (s.get('reference_id') is not None or s['semantic_preserving'] or
                s['family'] != 'baseline' or row['label'] != s['source_class']):
                raise ValueError('invalid labeled anchor')
        elif s.get('role') == 'variant':
            ref = lookup.get(s.get('reference_id'))
            rs = ref.get('stress', {}) if ref else {}
            if (not ref or rs.get('role') != 'anchor' or ref['label'] != s['source_class'] or
                rs.get('replicate') != s['replicate'] or rs.get('source_seed') != s['source_seed']):
                raise ValueError('variant must reference its own matching anchor')
            if s['semantic_preserving']:
                allowed = (s['family'] in ('density', 'rotation') or
                           (s['family'] == 'noise' and s['level'] == 'add_0.1') or
                           (s['family'] == 'translation' and s['level'] == 'x_0.05' and
                            s['source_class'] in ('center_cluster', 'localized_cluster', 'scratch') and
                            s['parameters'].get('dropped_points') == 0))
                if not allowed:
                    raise ValueError('this perturbation is exploratory under the v1 semantic policy')
                if row['label'] != ref['label']:
                    raise ValueError('preserving pair must have equal labels')
            elif row['label'] is not None:
                raise ValueError('ambiguous variants must have null labels')
        else:
            raise ValueError('unknown stress role')
    if len(suite_seeds) != 1 or not any(r['stress']['role'] == 'anchor' for r in rows):
        raise ValueError('one suite seed and at least one anchor required')


def summarize(predictions, labels):
    """Consistency includes invalid pairs in denominator; invalid/invalid never agrees."""
    by_id = {r['id']: r for r in predictions}
    pairs = []
    for row in predictions:
        s = row['stress']
        if not s['semantic_preserving']:
            continue
        reference = by_id[s['reference_id']]
        a, b = reference['prediction'], row['prediction']
        pairs.append(dict(reference_id=reference['id'], variant_id=row['id'],
                          family=s['family'], level=s['level'], source_class=s['source_class'],
                          seed=s['seed'], parameters=s['parameters'],
                          reference_prediction=a, variant_prediction=b,
                          both_valid=a in labels and b in labels,
                          consistent=a in labels and b in labels and a == b,
                          reference_correct=a == reference['label'],
                          variant_correct=b == row['label']))

    def pair_summary(items):
        valid = sum(p['both_valid'] for p in items)
        agreed = sum(p['consistent'] for p in items)
        correct_base = [p for p in items if p['reference_correct']]
        return dict(pairs=len(items), consistent_pairs=agreed,
                    consistency=agreed / len(items) if items else None,
                    both_valid_pairs=valid, valid_pair_coverage=valid / len(items) if items else None,
                    consistency_given_both_valid=agreed / valid if valid else None,
                    correct_reference_pairs=len(correct_base),
                    retained_accuracy_given_correct_reference=(sum(p['variant_correct'] for p in correct_base) /
                                                              len(correct_base) if correct_base else None))

    def group_report(items):
        labeled = [r for r in items if r['label'] is not None]
        exploratory = [r for r in items if r['label'] is None]
        ids = {r['id'] for r in items}
        return dict(samples=len(items), labeled_samples=len(labeled), exploratory_samples=len(exploratory),
                    labeled_metrics=metrics([r['label'] for r in labeled], [r['prediction'] for r in labeled],
                                            labels) if labeled else None,
                    consistency=pair_summary([p for p in pairs if p['variant_id'] in ids]),
                    exploratory_invalid_output_rate=(sum(r['prediction'] not in labels for r in exploratory) /
                                                     len(exploratory) if exploratory else None),
                    exploratory_prediction_counts=dict(Counter(
                        r['prediction'] if r['prediction'] in labels else '__invalid__' for r in exploratory)))

    families = sorted({r['stress']['family'] for r in predictions})
    return dict(schema=SCHEMA, real_fab_accuracy_validated=False,
                note='Synthetic sensitivity only. Constant wrong predictions can be perfectly consistent.',
                overall=group_report(predictions),
                anchors=group_report([r for r in predictions if r['stress']['role'] == 'anchor']),
                preserving_variants=group_report([r for r in predictions if r['stress']['semantic_preserving']]),
                by_family={f: group_report([r for r in predictions if r['stress']['family'] == f]) for f in families},
                by_family_level={f: {level: group_report([r for r in predictions if
                    r['stress']['family'] == f and r['stress']['level'] == level])
                    for level in sorted({r['stress']['level'] for r in predictions if r['stress']['family'] == f})}
                    for f in families},
                by_source_class={c: group_report([r for r in predictions if r['stress']['source_class'] == c])
                                 for c in labels}, pairs=pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--replicates', type=int, default=3)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error('--replicates must be positive')
    print(generate(args.output, args.seed, args.replicates))


if __name__ == '__main__':
    main()
