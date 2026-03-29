# Makefile
# Удобные команды для запуска pipeline

.PHONY: help prepare train build-graph build-attack-data build-attack-data-foreign run-attack clean

INPUT ?= data/[mod]_vae_3d_data.csv
SAMPLES ?= 43:12,44:13,58:13,87:14,96:11
TARGET_NCT ?= 2

# build-attack-data-foreign: один тестовый кадр (--photo) из NUM_FOREIGN чужих классов (исключая EXCLUDE_CLASS)
ATTACK_DATA_OUT ?= data/data_for_attack.csv
EXCLUDE_CLASS ?= 2
NUM_FOREIGN ?= 30
FOREIGN_PHOTO ?= 11
N_CLASSES ?= 200
N_TRAIN ?= 10
# Опционально: FOREIGN_SEED=0 make build-attack-data-foreign
FOREIGN_SEED ?=

# 0 = не останавливать по patience, откатать все --n-iterations (для «тяжёлых» латентов вроде mod VAE)
EARLY_STOPPING ?= 0

help:
	@echo "NCT Attack Framework - Available commands:"
	@echo "  make prepare            - Prepare input data"
	@echo "  make build-attack-data  - Build data_for_attack.csv from samples"
	@echo "  make build-attack-data-foreign  - Build CSV via --foreign-grid (override EXCLUDE_CLASS, NUM_FOREIGN, ...)"
	@echo "  make train              - Learn NCT model"
	@echo "  make build-graph        - Build correlation graph"
	@echo "  make run-attack         - Run attack algorithm"
	@echo "  make clean              - Clean build artifacts"

prepare:
	@echo "[*] Preparing data..."
	python ./python/prepare_data.py

train:
	@echo "[*] Training NCT model..."
# 	@mkdir -p model
	cd C#/NCT_cli && dotnet run -- train \
		--data ../../$(INPUT) \
		--output ../../model/meta.json \
		--classes 150 \
		--own-classes 10

build-graph:
	@echo "[*] Building correlation graph"
	python ./python/build_graph.py \
		--meta-path model/meta.json \
		--output-path model/graph.json \
		--nct-index $(TARGET_NCT) 

build-attack-data:	
	@echo "[*] Building data_for_attack.csv from $(INPUT), samples: $(SAMPLES)"
	python ./python/build_attack_data.py \
		--input $(INPUT) \
		--output $(ATTACK_DATA_OUT) \
		--samples "$(SAMPLES)"

build-attack-data-foreign:
	@echo "[*] Building $(ATTACK_DATA_OUT): foreign-grid, exclude=$(TARGET_NCT), n=$(NUM_FOREIGN), photo=$(FOREIGN_PHOTO)"
	python ./python/build_attack_data.py \
		--input $(INPUT) \
		--output $(ATTACK_DATA_OUT) \
		--foreign-grid \
		--exclude-class $(TARGET_NCT) \
		--num-foreign $(NUM_FOREIGN) \
		--photo $(FOREIGN_PHOTO)
# --n-classes $(N_CLASSES) \
# --n-train $(N_TRAIN) \
# $(if $(FOREIGN_SEED),--seed $(FOREIGN_SEED),)

# extract-synapses: 
# 	@echo "[*] Extracting synapses from model..."
# 	dotnet build C#/NCT_extraction/NCT_extraction.csproj -c Release
# 	./C#/NCT_extraction/bin/Release/net8.0/AttackCli extract-synapses \
# 		./model/model.bin \
# 		./model/synapses.json


run-attack:
	@echo "[*] Running attack framework... (EARLY_STOPPING=$(EARLY_STOPPING), 0=off)"
	cd C#/NCT_attack && dotnet run\
				--graph-json ../../model/graph.json \
				--model ../../model/meta.json \
				--input ../../$(ATTACK_DATA_OUT) \
				--output ../../results/samples \
				--learning-rate 0.005 \
				--step-size 0.5 \
				--n-iterations 1000 \
				--early-stopping $(EARLY_STOPPING) \
				--batch-size 0 \
				--target-nct $(TARGET_NCT)


# attack:
# 	@echo "[*] Running FGSM attack..."
# 	@cp config.yaml config_fgsm.yaml
# 	@sed -i 's/name: "identity"/name: "fgsm"/' config_fgsm.yaml
# 	python run_experiment.py config_fgsm.yaml

clean:
	@echo "[*] Cleaning up..."
	@rm -rf runs/*
	@rm -rf model/*
	@rm -rf data/cvae_3d_data_processed.csv
	@echo "[DONE]"


# .PHONY: all
# all: clean prepare train baseline
# 	@echo "Full pipeline complete"