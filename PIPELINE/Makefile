# Makefile
# Удобные команды для запуска pipeline

.PHONY: help prepare train build-graph run-attack clean

help:
	@echo "NCT Attack Framework - Available commands:"
	@echo "  make prepare            - Prepare input data"
	@echo "  make train              - Learn NCT model"
	@echo "  make build-graph        - Build correlation graph"
	@echo "  make run-attack         - Run attack algorithm"
	@echo "  make clean              - Clean build artifacts"

prepare:
	@echo "[*] Preparing data..."
# 	@mkdir -p data
	python ./python/prepare_data.py

train:
	@echo "[*] Training NCT model..."
# 	@mkdir -p model
	cd C#/NCT_cli && dotnet run -- train \
		--data ../../data/vae_3d_data.csv \
		--output ../../model/meta.json \
		--classes 150 \
		--own-classes 10

build-graph:
	@echo "[*] Building correlation graph"
	python ./python/nct_attack/build_graph.py \
		--meta-path model/meta.json \
		--output-path model/graph.json \
		--nct-index 0 


# extract-synapses: 
# 	@echo "[*] Extracting synapses from model..."
# 	dotnet build C#/NCT_extraction/NCT_extraction.csproj -c Release
# 	./C#/NCT_extraction/bin/Release/net8.0/AttackCli extract-synapses \
# 		./model/model.bin \
# 		./model/synapses.json


run-attack:
	@echo "[*] Running attack framework..."
	cd C#/NCT_attack && dotnet run\
				--graph-json ../../model/graph.json \
				--model ../../model/meta.json \
				--input ../../data/data_for_attack.csv \
				--output results/ \
				--learning-rate 0.005 \
				--step-size 1.0 \
				--n-iterations 100 \
				--early-stopping 20 \
				--batch-size 0 \
				--target-nct 0


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