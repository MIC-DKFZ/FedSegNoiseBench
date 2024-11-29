# Variables
PYTHON_VERSION = python3.10  # Specify the Python version
# VENV_DIR = .venv
VENVDIR = /dkfz/cluster/gpu/data/OE0441/m391k/venvs/.venv
REQ_FILE = requirements.txt
SUBMODULE_DIR = nnUNet

# Default target
.PHONY: all
all: setup

# Target to set up the virtual environment and install packages
.PHONY: setup
setup: $(VENV_DIR)/bin/activate
	$(VENV_DIR)/bin/pip install --upgrade pip setuptools wheel
	$(VENV_DIR)/bin/pip install -r $(REQ_FILE)
	$(VENV_DIR)/bin/pip install -e ./$(SUBMODULE_DIR)

# Create virtual environment if it doesn't exist
$(VENV_DIR)/bin/activate:
	module load python/3.10.1 && $(PYTHON_VERSION) -m venv $(VENV_DIR)

# Update packages from requirements.txt and submodule
.PHONY: update
update:
	module load python/3.10.1 && $(VENV_DIR)/bin/pip install --upgrade -r $(REQ_FILE)
	module load python/3.10.1 && $(VENV_DIR)/bin/pip install --upgrade -e ./$(SUBMODULE_DIR)

# Clean up the virtual environment
.PHONY: clean
clean:
	rm -rf $(VENV_DIR)
