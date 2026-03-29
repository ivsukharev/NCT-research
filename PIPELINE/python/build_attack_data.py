"""
Формирует data_for_attack.csv из *_3d_data.csv по списку образцов.

Использование:
  python build_attack_data.py --input data/vae_3d_data.csv --samples "4:12,6:13,8:13,11:14,19:11"

  По одному снимку из K чужих классов (тестовое фото), класс-жертва исключается:
  python build_attack_data.py --input data/[mod]_vae_3d_data.csv --foreign-grid \\
    --exclude-class 42 --num-foreign 100 --photo 11

Формат --samples: "person:photo,person:photo,..." (1-индексация)
  person: 1..200, photo: 1..14

--exclude-class: индекс класса как в CSV (0-based), тот же, что колонка class.
"""

import argparse
import random
from typing import List, Optional, Tuple

import numpy as np
from pathlib import Path


def parse_sample_pairs(samples: str, n_classes: int, n_per_class: int) -> List[Tuple[int, int]]:
    pairs = []
    for token in samples.split(','):
        token = token.strip()
        if not token:
            continue
        if ':' in token:
            p, f = token.split(':')
            person, photo = int(p), int(f)
            assert 1 <= person <= n_classes, f"person {person} вне [1,{n_classes}]"
            assert 1 <= photo <= n_per_class, f"photo {photo} вне [1,{n_per_class}]"
            pairs.append((person - 1, photo - 1))
        else:
            gid = int(token)
            pairs.append((gid // n_per_class, gid % n_per_class))
    return pairs


def build_foreign_pairs(
    exclude_class: int,
    num_foreign: int,
    photo_1based: int,
    n_classes: int,
    n_train: int,
    seed: Optional[int],
) -> List[Tuple[int, int]]:
    assert 0 <= exclude_class < n_classes, f"exclude_class {exclude_class} вне [0,{n_classes - 1}]"
    assert 1 <= photo_1based <= 14, "photo вне [1,14]"
    img_id = photo_1based - 1
    assert img_id >= n_train, (
        f"photo {photo_1based} попадает в train (n-train={n_train}); для атаки возьмите фото > {n_train}"
    )
    candidates = [c for c in range(n_classes) if c != exclude_class]
    if num_foreign > len(candidates):
        raise ValueError(f"num-foreign={num_foreign} больше числа чужих классов ({len(candidates)})")
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(candidates)
    chosen = candidates[:num_foreign]
    chosen.sort()
    return [(c, img_id) for c in chosen]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Путь к *_3d_data.csv (2800×512)')
    parser.add_argument('--output', default='data/data_for_attack.csv')
    parser.add_argument('--samples', default=None,
                        help='Образцы: "person:photo,..." (1-индексация)')
    parser.add_argument('--foreign-grid', action='store_true',
                        help='Один снимок (--photo) из --num-foreign классов, кроме --exclude-class')
    parser.add_argument('--exclude-class', type=int, default=None,
                        help='Индекс атакуемого класса (0-based, как колонка class), исключается из выборки')
    parser.add_argument('--num-foreign', type=int, default=100,
                        help='Сколько чужих классов (по одному фото)')
    parser.add_argument('--photo', type=int, default=11,
                        help='Номер фото 1..14 одинаково для всех выбранных классов (лучше из test: > n-train)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Для --foreign-grid: перемешать порядок классов (воспроизводимость)')
    parser.add_argument('--n-classes', type=int, default=200)
    parser.add_argument('--n-per-class', type=int, default=14)
    parser.add_argument('--n-train', type=int, default=10,
                        help='Первые N фото — train, остальные — test')
    args = parser.parse_args()

    if args.foreign_grid:
        if args.exclude_class is None:
            parser.error('--foreign-grid требует --exclude-class')
        pairs = build_foreign_pairs(
            exclude_class=args.exclude_class,
            num_foreign=args.num_foreign,
            photo_1based=args.photo,
            n_classes=args.n_classes,
            n_train=args.n_train,
            seed=args.seed,
        )
    else:
        if not args.samples:
            parser.error('Укажите --samples или --foreign-grid')
        pairs = parse_sample_pairs(args.samples, args.n_classes, args.n_per_class)

    data = np.loadtxt(args.input, delimiter=',')
    n_samples, n_features = data.shape
    expected = args.n_classes * args.n_per_class
    assert n_samples == expected, f"Ожидается {expected} строк, получено {n_samples}"

    header = 'id,class,split,' + ','.join(f'f{i}' for i in range(n_features))
    lines = [header]

    for class_id, img_id in pairs:
        gid = class_id * args.n_per_class + img_id
        split = 'train' if img_id < args.n_train else 'test'
        features = ','.join(str(v) for v in data[gid])
        lines.append(f'{gid},{class_id},{split},{features}')

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n')
    print(f"[DONE] {out}: {len(pairs)} образцов из {args.input}")
    for class_id, img_id in pairs:
        gid = class_id * args.n_per_class + img_id
        split = 'train' if img_id < args.n_train else 'test'
        print(f"  id={gid}, person={class_id+1}, photo={img_id+1}, split={split}")


if __name__ == '__main__':
    main()
