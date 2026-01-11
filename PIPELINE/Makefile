# Makefile
# Удобные команды для запуска pipeline

.PHONY: prepare train infer baseline attack clean help

help:
	@echo "NCT Adversarial Robustness Pipeline"
	@echo ""
	@echo "Usage:"
	@echo "  make prepare     - Переформатировать cvae_3d_data.csv (вариант B с метаданными)"
	@echo "  make train       - Обучить модель NCT (40 классов, 10 для обучения)"
	@echo "  make baseline    - Запустить baseline (identity attack)"
	@echo "  make attack      - Запустить FGSM атаку"
	@echo "  make clean       - Удалить все артефакты"
	@echo ""
	@echo "Examples:"
	@echo "  make prepare && make train && make baseline"
	@echo "  python run_experiment.py config.yaml"

prepare:
	@echo "[*] Preparing data..."
	@mkdir -p data
	python prepare_data.py

train:
	@echo "[*] Training NCT model..."
	@mkdir -p model
	cd csharp/NctCli && dotnet run -- train \
		--data ../../data/cvae_3d_data_processed.csv \
		--output ../../model/model.bin \
		--config ../../model/meta.json \
		--classes 40 \
		--own-classes 10

baseline:
	@echo "[*] Running baseline experiment (identity)..."
	python run_experiment.py config.yaml

attack:
	@echo "[*] Running FGSM attack..."
	@cp config.yaml config_fgsm.yaml
	@sed -i 's/name: "identity"/name: "fgsm"/' config_fgsm.yaml
	python run_experiment.py config_fgsm.yaml

clean:
	@echo "[*] Cleaning up..."
	@rm -rf runs/*
	@rm -rf model/*
	@rm -rf data/cvae_3d_data_processed.csv
	@echo "[✓] Done"

# Завод команд (полный цикл)
.PHONY: all
all: clean prepare train baseline
	@echo "[✓] Full pipeline complete!"

.PHONY: debug
debug:
	@echo "Config:"
	@cat config.yaml
	@echo ""
	@echo "Data shape:"
	@python -c "import pandas as pd; df = pd.read_csv('data/cvae_3d_data_processed.csv'); print(df.shape); print(df.head())"
