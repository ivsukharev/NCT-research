# run_experiment.py
# Orchestrator: читает CSV, генерирует атакованные примеры, вызывает infer, сохраняет результаты

import subprocess
import json
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import yaml

class AttackConfig:
    def __init__(self, name: str, **params):
        self.name = name
        self.params = params

    def to_dict(self):
        return {'name': self.name, **self.params}

class ExperimentRunner:
    def __init__(self, config_path: str):
        """Инициализация из YAML конфига"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.run_id = self.config.get('run_id', datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.run_dir = Path(self.config.get('output_dir', 'runs')) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        print(f"[*] Run ID: {self.run_id}")
        print(f"[*] Output: {self.run_dir}")

    def load_data(self, csv_path: str) -> List[Tuple[int, int, List[float]]]:
        """Загрузка CSV в формате: id,class,split,f0..f511"""
        print(f"[*] Loading data from {csv_path}...")
        data = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_id = int(row['id'])
                class_label = int(row['class'])
                features = [float(row[f'f{i}']) for i in range(512)]
                data.append((sample_id, class_label, features))

        print(f"    Loaded {len(data)} samples")
        return data

    def attack_identity(self, data: List[Tuple[int, int, List[float]]]) -> Tuple[List, Dict]:
        """Identity attack: no perturbation"""
        print(f"[ATTACK] Running identity attack...")
        attacked = [features[:] for _, _, features in data]  # copy
        return attacked, {'num_queries': 0, 'norms': [0.0] * len(attacked)}

    def attack_fgsm(self, data: List[Tuple[int, int, List[float]]], epsilon: float = 0.01, 
                    norm: str = 'l2') -> Tuple[List, Dict]:
        """FGSM-like attack: random perturbation"""
        print(f"[ATTACK] Running FGSM attack (eps={epsilon}, norm={norm})...")
        attacked = []
        norms = []

        for sample_id, class_label, features in data:
            # Случайное возмущение
            delta = np.random.randn(512) * epsilon
            x_adv = np.array(features) + delta

            # Clip into valid range (опционально)
            x_adv = np.clip(x_adv, np.array(features) - epsilon * 3, np.array(features) + epsilon * 3)

            attacked.append(x_adv.tolist())
            
            if norm == 'l2':
                norms.append(float(np.linalg.norm(delta)))
            elif norm == 'linf':
                norms.append(float(np.max(np.abs(delta))))

        return attacked, {'num_queries': len(data), 'norms': norms}

    def run_inference(self, input_csv: str, output_json: str):
        """Вызов C# infer CLI"""
        print(f"[*] Running inference...")
        cmd = [
            'dotnet', 'run', '--project', self.config['infer_cli_path'],
            '--', 'infer',
            '--model', self.config['model_bin'],
            '--meta', self.config['model_meta'],
            '--input', input_csv,
            '--output', output_json
        ]

        print(f"    Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[ERROR] Inference failed!")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            raise RuntimeError("Inference failed")

        print(f"    Inference complete")

    def export_csv(self, data: List[Tuple[int, int, List[float]]], csv_path: str):
        """Сохранить данные в CSV для infer"""
        print(f"[*] Exporting to {csv_path}...")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Заголовок
            header = ['id', 'class', 'split'] + [f'f{i}' for i in range(512)]
            writer.writerow(header)

            for sample_id, class_label, features in data:
                row = [sample_id, class_label, 'attack'] + features
                writer.writerow(row)

    def compute_metrics(self, pred_clean_json: str, pred_adv_json: str) -> Dict:
        """Сравнить чистые и атакованные предсказания"""
        print(f"[*] Computing metrics...")

        with open(pred_clean_json, 'r') as f:
            pred_clean = json.load(f)
        with open(pred_adv_json, 'r') as f:
            pred_adv = json.load(f)

        predictions_clean = {p['id']: p for p in pred_clean['predictions']}
        predictions_adv = {p['id']: p for p in pred_adv['predictions']}

        # Основные метрики
        success_count = 0
        avg_hamming_clean = 0.0
        avg_hamming_adv = 0.0
        misclassified_count = 0

        for sample_id in predictions_clean.keys():
            clean = predictions_clean[sample_id]
            adv = predictions_adv[sample_id]

            avg_hamming_clean += clean['best_hamming']
            avg_hamming_adv += adv['best_hamming']

            # Attack success: класс изменился
            if clean['pred_class'] != adv['pred_class']:
                success_count += 1
                misclassified_count += 1

        n = len(predictions_clean)
        metrics = {
            'attack_success_rate': success_count / n if n > 0 else 0,
            'misclassified_count': misclassified_count,
            'avg_hamming_clean': avg_hamming_clean / n if n > 0 else 0,
            'avg_hamming_adv': avg_hamming_adv / n if n > 0 else 0,
            'total_samples': n
        }

        return metrics

    def run(self):
        """Запустить полный pipeline"""
        print("=" * 60)
        print(f"NCT Adversarial Robustness Pipeline")
        print("=" * 60)

        # 1. Загружаем данные
        data = self.load_data(self.config['data_csv'])

        # 2. Выполняем инференс на чистых данных
        print(f"\n[PHASE 1] Clean inference baseline...")
        clean_csv = self.run_dir / 'input_clean.csv'
        pred_clean_json = self.run_dir / 'pred_clean.json'

        self.export_csv(data, str(clean_csv))
        self.run_inference(str(clean_csv), str(pred_clean_json))

        # 3. Выполняем атаку
        print(f"\n[PHASE 2] Adversarial attack...")
        attack_config = AttackConfig(
            self.config['attack']['name'],
            **self.config['attack'].get('params', {})
        )

        if attack_config.name == 'identity':
            attacked, attack_stats = self.attack_identity(data)
        elif attack_config.name == 'fgsm':
            attacked, attack_stats = self.attack_fgsm(
                data,
                epsilon=attack_config.params.get('epsilon', 0.01),
                norm=attack_config.params.get('norm', 'l2')
            )
        else:
            raise ValueError(f"Unknown attack: {attack_config.name}")

        # 4. Экспортируем атакованные примеры
        attacked_data = [
            (sample_id, class_label, attacked_features)
            for (sample_id, class_label, _), attacked_features in zip(data, attacked)
        ]

        adv_csv = self.run_dir / 'input_adv.csv'
        pred_adv_json = self.run_dir / 'pred_adv.json'

        self.export_csv(attacked_data, str(adv_csv))

        # 5. Выполняем инференс на атакованных данных
        print(f"\n[PHASE 3] Adversarial inference...")
        self.run_inference(str(adv_csv), str(pred_adv_json))

        # 6. Считаем метрики
        print(f"\n[PHASE 4] Computing metrics...")
        metrics = self.compute_metrics(str(pred_clean_json), str(pred_adv_json))

        # 7. Сохраняем результаты
        results = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'data': self.config['data_csv'],
                'attack': attack_config.to_dict(),
                'model': self.config['model_bin']
            },
            'attack_stats': attack_stats,
            'metrics': metrics,
            'files': {
                'input_clean': str(clean_csv),
                'pred_clean': str(pred_clean_json),
                'input_adv': str(adv_csv),
                'pred_adv': str(pred_adv_json)
            }
        }

        results_json = self.run_dir / 'results.json'
        with open(results_json, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n" + "=" * 60)
        print(f"RESULTS:")
        print(f"  Attack success rate: {metrics['attack_success_rate']:.2%}")
        print(f"  Misclassified samples: {metrics['misclassified_count']}/{metrics['total_samples']}")
        print(f"  Avg Hamming (clean): {metrics['avg_hamming_clean']:.2f}")
        print(f"  Avg Hamming (adv): {metrics['avg_hamming_adv']:.2f}")
        print(f"\nArtifacts saved to: {self.run_dir}")
        print(f"Full results: {results_json}")
        print("=" * 60)

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python run_experiment.py <config.yaml>")
        sys.exit(1)

    runner = ExperimentRunner(sys.argv[1])
    runner.run()
