all: run_models
	
run_models:
	python3 scripts/DoS_SMOTE_ML.py > results/model_results.txt

validate_smote_k_train:
	python3 scripts/DoS_SMOTE_ML.py > results/validation_results.txt

feature_selection:
	python3 scripts/DoS_SMOTE_ML.py > results/feature_selection_results.txt
	
combine_data:
	scripts/combine_csv.sh 02-15-2018.csv 02-16-2018.csv 02-21-2018.csv

clean:
	rm -f results.txt