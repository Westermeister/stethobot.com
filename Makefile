.PHONY: all
all: black prettier styles

.PHONY: black
black:
	python -m black stethobot

.PHONY: prettier
prettier:
	npx prettier --write "./*.json" "./stethobot"

.PHONY: styles
styles: prettier
	npx sass stethobot/static/styles/sass:stethobot/static/styles/css --style compressed --no-source-map
