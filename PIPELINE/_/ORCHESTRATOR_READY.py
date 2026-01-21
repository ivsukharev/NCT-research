# orchestrator.py (переименовать из run_attack.py)
# Находится в python/nct-attack/

```python
"""
ОРКЕСТРАТОР: Управление всеми этапами атаки

Этот модуль координирует запуск трёх этапов:
  1. Построение графа корреляций
  2. Аугментация изображений
  3. Атака на основе градиентного спуска

Использование:
    # Запустить все этапы
    python orchestrator.py

    # Запустить только конкретные этапы
    python orchestrator.py --stages 1 3

    # В коде
    from nct_attack.orchestrator import run_full_pipeline
    results = run_full_pipeline(stages=[1, 2, 3])
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

from nct_attack.logger import get_logger
from nct_attack.stages.stage_01_build_graph import stage_01_build_graph
from nct_attack.stages.stage_02_augment import stage_02_augment
from nct_attack.stages.stage_03_attack import stage_03_attack

logger = get_logger(__name__)


class PipelineOrchestrator:
    """Класс для управления этапами конвейера"""
    
    def __init__(
        self,
        meta_json_path: str = "model/meta.json",
        nct_index: int = 0,
    ):
        """
        Args:
            meta_json_path: путь к meta.json
            nct_index: индекс NCT
        """
        self.meta_json_path = meta_json_path
        self.nct_index = nct_index
        self.results = {}
    
    def run(self, stages: List[int] = None) -> Dict[str, Any]:
        """
        Запустить конвейер
        
        Args:
            stages: список этапов [1, 2, 3]
        
        Returns:
            dict с результатами всех этапов
        """
        if stages is None:
            stages = [1, 2, 3]
        
        # Валидация этапов
        stages = [s for s in stages if s in [1, 2, 3]]
        if not stages:
            logger.error("❌ Не указаны корректные этапы (1, 2, 3)")
            return {}
        
        logger.info("")
        logger.info("╔" + "=" * 68 + "╗")
        logger.info("║" + " " * 68 + "║")
        logger.info("║" + "  🚀 ЗАПУСК КОНВЕЙЕРА АТАКИ".center(68) + "║")
        logger.info("║" + f"  Этапы: {stages}".ljust(68) + "║")
        logger.info("║" + " " * 68 + "║")
        logger.info("╚" + "=" * 68 + "╝")
        logger.info("")
        
        try:
            # Этап 1: Построение графа
            if 1 in stages:
                logger.info("▶ Переход к Этапу 1...")
                self.results["stage_1"] = self._run_stage_1()
            
            # Этап 2: Аугментация
            if 2 in stages:
                logger.info("▶ Переход к Этапу 2...")
                graph_path = self.results.get("stage_1", {}).get(
                    "graph_path", 
                    "model/graph.json"
                )
                self.results["stage_2"] = self._run_stage_2(graph_path)
            
            # Этап 3: Атака
            if 3 in stages:
                logger.info("▶ Переход к Этапу 3...")
                graph_path = self.results.get("stage_1", {}).get(
                    "graph_path",
                    "model/graph.json"
                )
                augment_path = self.results.get("stage_2", {}).get(
                    "augmentation_path",
                    "model/augmentation_data.json"
                )
                self.results["stage_3"] = self._run_stage_3(graph_path, augment_path)
            
            # Итоговый отчет
            self._print_summary(stages)
            
            return self.results
            
        except KeyboardInterrupt:
            logger.warning("\n⚠️  Конвейер прерван пользователем")
            sys.exit(1)
        
        except Exception as e:
            logger.error(f"\n❌ Критическая ошибка в конвейере: {e}", exc_info=True)
            sys.exit(1)
    
    def _run_stage_1(self) -> Dict[str, Any]:
        """Запустить этап 1"""
        try:
            return stage_01_build_graph(
                meta_json_path=self.meta_json_path,
                nct_index=self.nct_index,
            )
        except Exception as e:
            logger.error(f"Ошибка на этапе 1: {e}")
            raise
    
    def _run_stage_2(self, graph_path: str) -> Dict[str, Any]:
        """Запустить этап 2"""
        try:
            return stage_02_augment(graph_json_path=graph_path)
        except Exception as e:
            logger.error(f"Ошибка на этапе 2: {e}")
            raise
    
    def _run_stage_3(self, graph_path: str, augment_path: str) -> Dict[str, Any]:
        """Запустить этап 3"""
        try:
            return stage_03_attack(
                graph_json_path=graph_path,
                augment_data_path=augment_path,
            )
        except Exception as e:
            logger.error(f"Ошибка на этапе 3: {e}")
            raise
    
    def _print_summary(self, stages: List[int]) -> None:
        """Вывести итоговый отчет"""
        logger.info("")
        logger.info("╔" + "=" * 68 + "╗")
        logger.info("║" + " " * 68 + "║")
        logger.info("║" + "  ✅ КОНВЕЙЕР УСПЕШНО ЗАВЕРШЁН".center(68) + "║")
        logger.info("║" + " " * 68 + "║")
        
        # Информация по этапам
        for stage_num in stages:
            stage_key = f"stage_{stage_num}"
            if stage_key in self.results:
                logger.info("║" + f"  ✓ Этап {stage_num}: Завершён".ljust(68) + "║")
        
        logger.info("║" + " " * 68 + "║")
        logger.info("╚" + "=" * 68 + "╝")
        logger.info("")


def run_full_pipeline(
    meta_json_path: str = "model/meta.json",
    nct_index: int = 0,
    stages: List[int] = None,
) -> Dict[str, Any]:
    """
    Функция-обёртка для запуска полного конвейера
    
    Args:
        meta_json_path: путь к meta.json
        nct_index: индекс NCT
        stages: список этапов [1, 2, 3]
    
    Returns:
        dict с результатами
    """
    orchestrator = PipelineOrchestrator(
        meta_json_path=meta_json_path,
        nct_index=nct_index,
    )
    return orchestrator.run(stages=stages)


def main():
    """Запуск как скрипт"""
    parser = argparse.ArgumentParser(
        description="Оркестратор конвейера атаки NCT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python orchestrator.py                    # Все этапы (1, 2, 3)
  python orchestrator.py --stages 1         # Только этап 1
  python orchestrator.py --stages 1 3       # Этапы 1 и 3
  python orchestrator.py --stages 1 2 3     # Явно все этапы
        """
    )
    
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Этапы для запуска (по умолчанию: 1 2 3)"
    )
    
    parser.add_argument(
        "--meta-path",
        type=str,
        default="model/meta.json",
        help="Путь к meta.json (по умолчанию: model/meta.json)"
    )
    
    parser.add_argument(
        "--nct-index",
        type=int,
        default=0,
        help="Индекс NCT в массиве (по умолчанию: 0)"
    )
    
    args = parser.parse_args()
    
    # Запустить конвейер
    results = run_full_pipeline(
        meta_json_path=args.meta_path,
        nct_index=args.nct_index,
        stages=args.stages,
    )
    
    # Вывести результаты
    if results:
        logger.info("\n📊 Результаты конвейера:")
        for stage_key, stage_result in results.items():
            logger.info(f"  {stage_key}: ✓")


if __name__ == "__main__":
    main()
```

---

## 📌 ИСПОЛЬЗОВАНИЕ:

### Запуск как скрипт:
```bash
# Все этапы (по умолчанию)
python nct_attack/orchestrator.py

# Только этап 1
python nct_attack/orchestrator.py --stages 1

# Этапы 1 и 3
python nct_attack/orchestrator.py --stages 1 3

# С кастомным meta.json
python nct_attack/orchestrator.py --meta-path data/my_meta.json --stages 1
```

### Использование в коде:
```python
from nct_attack.orchestrator import run_full_pipeline

# Запустить все этапы
results = run_full_pipeline()

# Запустить только этап 1
results = run_full_pipeline(stages=[1])

# Проверить результаты
print(results["stage_1"]["feature_count"])
```

### С Makefile:
```bash
make stage1       # Только этап 1
make stage2       # Только этап 2
make stage3       # Только этап 3
make attack       # Все этапы подряд
```

---

## 🎯 ПРЕИМУЩЕСТВА:

✓ **Модульный** — каждый этап независим  
✓ **Гибкий** — можно запускать отдельные этапы  
✓ **Отказоустойчивый** — обработка ошибок  
✓ **Логируемый** — подробная информация  
✓ **Масштабируемый** — легко добавлять этапы  
✓ **CLI-friendly** — удобно из терминала  

Готово к использованию! 🎉
