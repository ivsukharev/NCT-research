# NCT Adversarial Robustness Testing Pipeline

Полный pipeline для тестирования устойчивости нейронной сети коррелционных преобразователей (NCT) из ГОСТ Р к состязательным атакам.

## Структура

```
.
├─ data/
│  └─ cvae_3d_data.csv             # исходные 512-мерные признаки (2800×512)
│  └─ cvae_3d_data_processed.csv   # переформатированные данные (вариант B)
├─ model/
│  ├─ model.bin                    # сериализованная обученная модель NCT
│  └─ meta.json                    # метаданные модели
├─ csharp/
│  └─ NctCli/
│     └─ NctCli.csproj             # C# проект train/infer CLI
├─ python/
│  ├─ prepare_data.py              # переформатирование CSV
│  ├─ run_experiment.py            # orchestrator атак
│  └─ config.yaml                  # конфигурация эксперимента
├─ runs/
│  └─ <run_id>/
│     ├─ input_clean.csv           # чистые примеры для baseline
│     ├─ pred_clean.json           # предсказания на чистых данных
│     ├─ input_adv.csv             # атакованные примеры
│     ├─ pred_adv.json             # предсказания на атакованных данных
│     └─ results.json              # итоговые метрики
└─ Makefile
```

## Быстрый старт

### 1. Подготовка данных (вариант B с метаданными)
```bash
make prepare
# или
python prepare_data.py
```

Создаёт `data/cvae_3d_data_processed.csv` с колонками:
- `id` — уникальный идентификатор образца (0–2799)
- `class` — класс (0–199)
- `split` — тип (train/test)
- `f0..f511` — 512 признаков

### 2. Обучение модели NCT
```bash
make train
# или вручную:
cd csharp/NctCli
dotnet run -- train \
  --data ../../data/cvae_3d_data_processed.csv \
  --output ../../model/model.bin \
  --config ../../model/meta.json \
  --classes 40 \
  --own-classes 10
```

Обучает 10 NCT на первых 40 классах датасета FEI Face.  
Сохраняет:
- `model/model.bin` — веса, пороги, таблицы активации, `sxstranger`, ключи
- `model/meta.json` — метаданные (version, feature_count, bit_per_neuron и т.д.)

### 3. Запуск базового теста (identity attack — без возмущений)
```bash
make baseline
# или
python run_experiment.py config.yaml
```

Выполняет:
1. Инференс на чистых данных → `pred_clean.json`
2. Копирует данные (атака identity) → `input_adv.csv`
3. Инференс на "атакованных" (идентичных) данных → `pred_adv.json`
4. Считает метрики и сохраняет в `runs/<run_id>/results.json`

### 4. Запуск FGSM атаки
```bash
make attack
# или вручную отредактируйте config.yaml и измените attack.name на "fgsm":
python run_experiment.py config.yaml
```

## Формат вывода

### pred_clean.json / pred_adv.json
```json
{
  "model_version": 1,
  "feature_count": 512,
  "own_classes": 10,
  "timestamp": "2026-01-11T15:34:00",
  "predictions": [
    {
      "id": 0,
      "true_class": 0,
      "pred_class": 0,
      "best_hamming": 23,
      "bit_array": "0101...1100"
    },
    ...
  ]
}
```

**Поля:**
- `pred_class` — выбранный класс (0–9), соответствующий NCT с минимальным расстоянием Хэмминга
- `best_hamming` — расстояние Хэмминга между кодом и ключом выбранного NCT
- `bit_array` — полный BitArray (256 бит = 2 бита × 128 нейронов) для диагностики

### results.json
```json
{
  "run_id": "baseline_identity",
  "timestamp": "2026-01-11T15:34:00",
  "config": {
    "data": "data/cvae_3d_data_processed.csv",
    "attack": {
      "name": "identity",
      "epsilon": null,
      "norm": null
    },
    "model": "model/model.bin"
  },
  "attack_stats": {
    "num_queries": 2800,
    "norms": [0.0, 0.0, ...]
  },
  "metrics": {
    "attack_success_rate": 0.0,
    "misclassified_count": 0,
    "avg_hamming_clean": 45.3,
    "avg_hamming_adv": 45.3,
    "total_samples": 2800
  },
  "files": {
    "input_clean": "runs/baseline_identity/input_clean.csv",
    "pred_clean": "runs/baseline_identity/pred_clean.json",
    "input_adv": "runs/baseline_identity/input_adv.csv",
    "pred_adv": "runs/baseline_identity/pred_adv.json"
  }
}
```

## Расширение: добавление собственных атак

Отредактируйте `run_experiment.py`, добавьте метод в класс `ExperimentRunner`:

```python
def attack_my_custom(self, data, **params):
    """Моя атака"""
    print(f"[ATTACK] Running custom attack...")
    attacked = []
    norms = []
    
    for sample_id, class_label, features in data:
        x_adv = np.array(features)
        # ваша логика
        attacked.append(x_adv.tolist())
        norms.append(...)
    
    return attacked, {'num_queries': ..., 'norms': norms}
```

И добавьте в `run()`:
```python
elif attack_config.name == 'my_custom':
    attacked, attack_stats = self.attack_my_custom(data, **attack_config.params)
```

Затем в `config.yaml`:
```yaml
attack:
  name: "my_custom"
  params:
    param1: value1
```

## Рекомендации для исследований

### Базовые атаки (для проверки)
1. **identity** — no perturbation, должен показать baseline accuracy
2. **fgsm** — случайные возмущения, epsilon∈[0.001, 0.1]

### Измеряемые метрики
- **Attack Success Rate** = (количество примеров, где `pred_adv ≠ pred_clean`) / total
- **Avg Hamming Distance** — средняя близость кода к ключу
- **Perturbation Norm** (L2/L∞) — величина возмущения в исходном пространстве

### Диагностика через BitArray
Сохранённый `bit_array` позволяет анализировать, какие биты меняются при атаке:
- Если только несколько бит флиппаются → нейроны чувствительны к конкретным направлениям
- Если почти все биты флиппаются → распределённая уязвимость

## Зависимости

### Python
```bash
pip install pyyaml pandas numpy
```

### C# (.NET 6+)
```bash
dotnet add package Newtonsoft.Json
```

## Логирование и отладка

Все логи выводятся в stdout. Для сохранения:
```bash
python run_experiment.py config.yaml 2>&1 | tee runs/$(date +%Y%m%d_%H%M%S)/log.txt
```

Для отладки конкретного примера:
```python
# В run_experiment.py добавьте:
print(f"[DEBUG] Sample {sample_id}: features_norm={np.linalg.norm(features)}")
print(f"[DEBUG] Sample {sample_id}: pred_clean={clean['pred_class']}, pred_adv={adv['pred_class']}")
```

## Производительность

- **Подготовка данных**: ~ 5 сек
- **Обучение (40 классов, 10 NCT)**: ~ 10–30 сек
- **Инференс (2800 примеров)**: ~ 2–5 сек (зависит от количества NCT)
- **FGSM атака (2800 примеров)**: ~ 1 сек

Полный цикл (подготовка + обучение + baseline): ~ 1 минута

## Контроль версий

```bash
git init
git add .
git commit -m "Initial commit: NCT robustness pipeline"
```

Версионировать:
- `model/model.bin` и `model/meta.json` (используйте DVC или Git LFS для больших файлов)
- `config.yaml` (разные конфиги для разных экспериментов)
- `runs/<run_id>/` (сохраняет каждый эксперимент отдельно)

## Ссылки

- [ГОСТ Р: Защищенные нейросетевые алгоритмы классификации](docs/GOST_R_2022.pdf)
- [Состязательные атаки на нейронные сети](docs/adversarial_attacks.pdf)
