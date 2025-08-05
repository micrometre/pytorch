.PHONY: install train prediction clean
# Installation and setup
install:
	pip install -r requirements.txt
# Clean generated files
clean:
	rm -rf data