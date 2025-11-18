install:
	pip install --upgrade pip
	pip install -r requirements.txt

format:
	black train.py App/app.py

train:
	python train.py

eval:
	echo "## Model Performance" > report.md
	echo "" >> report.md
	cat Results/metrics.txt >> report.md
	echo "" >> report.md
	echo "![Model Results](./Results/model_results.png)" >> report.md
	cml comment create report.md

update-branch:
	git config --global user.name "$(USER_NAME)"
	git config --global user.email "$(USER_EMAIL)"
	git add Results/metrics.txt Results/model_results.png Model/model_pipeline.skops
	git commit -m "Update model results and metrics [skip ci]" || echo "No changes to commit"
	git push origin HEAD:$(git rev-parse --abbrev-ref HEAD) || echo "Push failed"

deploy:
	pip install huggingface-hub
	python -c "from huggingface_hub import HfApi; \
HfApi().create_repo(repo_id='yahyasd56/house-price-predictor', repo_type='space', space_sdk='gradio', token='$(HF)', exist_ok=True); \
HfApi().upload_folder(folder_path='App', path_in_repo='.', repo_id='yahyasd56/house-price-predictor', repo_type='space', token='$(HF)')"

clean:
	rm -rf __pycache__ .pytest_cache
	find . -type f -name "*.pyc" -delete

test:
	pytest

all: install format train eval

.PHONY: install format train eval update-branch deploy clean test all
