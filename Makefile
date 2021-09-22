.PHONY: all
all: black models prettier pyflakes styles

.PHONY: check
check:
	python -m pytest tests --capture=fd

.PHONY: clean
clean:
	rm -f ./stethobot/api/breast_cancer/model.json
	rm -f ./stethobot/api/diabetes/model.json
	rm -f ./stethobot/api/heart_disease/model.json

.PHONY: black
black:
	python -m black stethobot

.PHONY: models
models:
	cd ./stethobot/api/breast_cancer && python fit.py
	cd ./stethobot/api/diabetes && python fit.py
	cd ./stethobot/api/heart_disease && python fit.py

.PHONY: prettier
prettier:
	npx prettier --write "./*.json" "./stethobot"

.PHONY: pyflakes
pyflakes:
	python -m pyflakes ./stethobot

.PHONY: styles
styles: prettier
	npx sass stethobot/static/styles/sass:stethobot/static/styles/css --style compressed --no-source-map
