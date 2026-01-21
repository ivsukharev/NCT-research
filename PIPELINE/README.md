# NCT Adversarial Pipeline

Пайплайн для подготовки данных, обучения NCT и запуска атак с учётом графа корреляций.

## Требования к ПО

- **Python 3.9+** : с пакетами: `numpy pandas pyyaml`

- **.NET SDK 8.0 (или 6.0+)**: для запуска C#-проектов


- **Make** (для запуска через Makefile, необязательно)  
  - Windows: установить `make` например через Chocolatey:  `choco install make`
  - Linux: `sudo apt-get install make`  

## Структура
```
PIPELINE/
├─ Makefile                      # prepare / train / build-graph / run-attack
├─ data/                         # CSV для обучения/атаки
│   ├─ cvae_3d_data.csv
│   ├─ vae_3d_data.csv
│   └─ data_for_attack.csv
├─ model/                        
│   ├─ meta.json                 # сохранённая обученная модель
│   ├─ graph.json                # граф корреляций для атаки
│   ├─ _cvae_meta.json           # альтернативная модель, обученаая на CVAE-признаках
│   └─ _cvae_graph.json          # граф корреляций CVAE-признаков
├─ C#/                          
│   ├─ NCT_framework/            # фреймворк NCT из стандарта
│   ├─ NCT_cli/                  # выполняет train 
│   └─ NCT_attack/               # атака на модель с учётом графа корреляций
└─ python/
    ├─ prepare_data.py           # подготовка CSV (разметка)
    └─ nct_attack/
        ├─ build_graph.py        # построение графа корреляций
        └─ logger.py
```

## Порядок запуска

1) Подготовить данные  
```bash
make prepare
```
Готовит `data/cvae_3d_data_processed.csv` или `data/vae_3d_data_processed.csv` (512 признаков).

2) Обучить NCT-модель  
```bash
make train
```
Сохраняет в `model/meta.json`.

3) Построить граф корреляций для атаки  
```bash
make build-graph
```
Сохраняет в`model/graph.json`.

4) Запустить атаку (C#)  
```bash
make run-attack
```
По умолчанию:
- `graph`: `model/graph.json`
- `model`: `model/meta.json`
- `input`: `data/data_for_attack.csv`
- `output`: `C#/NCT_attack/results/`

Ключевые параметры CLI (см. `C#/NCT_attack/NCT_attack.cs`), можно менять в Makefile:
- `--learning-rate` (default 0.01)
- `--step-size` (default 1.0)
- `--n-iterations` (default 100)
- `--early-stopping` (default 30) — сколько итераций максимально может быть проделано без улучшения
- `--batch-size` (0 = весь входящий датасет)
- `--target-nct` (индекс целевого NCT)


Результаты атаки в `C#/NCT_attack/results/`