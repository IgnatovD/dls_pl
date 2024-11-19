install:
	python3 -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -r requirements.txt

train:
	PYTHONPATH=. python src/train.py configs/config.yaml
