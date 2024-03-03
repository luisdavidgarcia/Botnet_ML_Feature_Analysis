all: run_models
	
run_models:
	python3 DoS_SMOTE_ML.py > results.txt

clean:
	rm -f results.txt