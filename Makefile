all: run_models
	
run_models:
	python3 scripts/DoS_SMOTE_ML.py > results/model_results.txt

validate_smote_k_train:
	python3 scripts/DoS_SMOTE_ML.py > results/validation_results.txt
	

clean:
	rm -f results.txt