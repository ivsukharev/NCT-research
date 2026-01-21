# prepare_data.py
# Переформатирование cvae_3d_data.csv в вариант B с метаданными

import pandas as pd
import numpy as np
from pathlib import Path

def prepare_cvae_dataset(csv_path: str, output_path: str):

    print(f"[*] Загрузка {csv_path}...")
    data = np.loadtxt(csv_path, delimiter=',')
    print(f"    Shape: {data.shape}")
    
    n_samples, n_features = data.shape
    
    n_classes = 200
    n_per_class = 14
    assert n_samples == n_classes * n_per_class, \
        f"Ожидается {n_classes}×{n_per_class}={n_classes*n_per_class} образцов, получено {n_samples}"
    
    # Создаём DataFrame
    records = []
    idx = 0
    for class_id in range(n_classes):
        for img_id in range(n_per_class):
            record = {
                'id': idx,
                'class': class_id,
                'split': 'train' if img_id < 10 else 'test',  # первые 10 обучение, 4 тест
                **{f'f{i}': data[idx, i] for i in range(n_features)}
            }
            records.append(record)
            idx += 1
    
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"[DONE] Сохранено в {output_path}...")
    print(f"    ")
    print(df.head(3))
    print(f"...")

if __name__ == '__main__':
    prepare_cvae_dataset(
        csv_path='data/vae_3d_data.csv',
        output_path='data/vae_3d_data_processed.csv'
    )
