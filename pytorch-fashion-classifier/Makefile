.PHONY: install train test predict clean help setup data
# Installation and setup
install:
	pip install -r requirements.txt
	python -m pip install -U pip
	python -m pip install -U matplotlib


# Create necessary directories
setup:
	mkdir -p data models 

# Dowcnload and prepare data (included in training)
data:
	@echo "Data will be downloaded automatically during training"

# Training
train: setup
	cd src && python train.py

# Testing
test:
	cd src && python test.py

# Predictions
predict:
	cd src && python predict.py
# Full workflow: install dependencies, train model, and test it
all: install train test

# Clean generated files
clean:
	rm -rf data
	rm -rf models/*.pth

# Clean everything including downloaded data
clean-all: clean
	rm -rf __pycache__ src/__pycache__

# Help target
help:
	@echo "Available targets:"
	@echo "  install   - Install project dependencies"
	@echo "  setup     - Create necessary directories"
	@echo "  train     - Train the fashion classification model"
	@echo "  test      - Test the trained model on test dataset"
	@echo "  predict   - Make predictions on custom images in test_images/"
	@echo "  all       - Run complete workflow (install, train, test)"
	@echo "  clean     - Remove generated data and model files"
	@echo "  clean-all - Remove all generated files including cache"
	@echo "  help      - Show this help message"