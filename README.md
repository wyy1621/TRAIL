# TRAIL

## Environment

Requires Python 3.12, PyTorch with CUDA support, and the system command bgpdump 1.6.2.

    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    sudo apt-get update
    sudo apt-get install -y bgpdump
    bgpdump -V

## Main Entry Points

Run the complete pipeline:

    bash run_pipeline.sh

Or run the stages manually:

    python download_data.py
    python build_dataset.py
    python train_reconstructor.py
    python train_detector.py
    python evaluate.py

The scripts use config.yaml by default. Add --config <path> to use another configuration file.
