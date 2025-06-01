# LLM from Scratch - Demo Scripts 🚀

This folder contains scripts and notebooks that provide educational walkthroughs and visualizations related to the concepts covered in the book "Build a Large Language Model (From Scratch)".

## 🛠️ Setup

Before running these scripts, please ensure you have set up your Python environment and installed the necessary dependencies as described in the main [setup guide](../../setup/README.md) for this repository. This project uses `uv` for managing environments and dependencies.

To set up the environment and install dependencies using `uv`, navigate to the root of the repository and run:
```bash
# Create a virtual environment (if you haven't already)
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate

# Install dependencies using uv
uv pip install -r requirements.txt
```
✅ Ensure your `uv`-managed virtual environment is activated before running the scripts.

## ሩጫ Running the Demo Scripts

The following Python scripts offer visualizations and training examples:

### 1. `gpt_visual.py` 🖼️

This script likely provides a visualization of the GPT model's architecture or its components.

To run it (ensure your virtual environment is active):
```bash
python demo/gpt_visual.py
```
Or, if you have `uv run` configured for your project (e.g., via `pyproject.toml` scripts):
```bash
uv run python demo/gpt_visual.py
```

### 2. `gpt_visual_train.py` 🏋️‍♀️

This script is probably used to train a version of the GPT model, possibly with visual outputs or for a smaller dataset suitable for demonstration.

To run it (ensure your virtual environment is active):
```bash
python demo/gpt_visual_train.py
```
Or, using `uv run`:
```bash
uv run python demo/gpt_visual_train.py
```
📝 You may need to check the script for specific command-line arguments or data requirements.

### 3. `gpt_visual_attention.py` 👀

This script likely visualizes the attention mechanism within the GPT model, showing how different parts of the input sequence attend to each other.

To run it (ensure your virtual environment is active):
```bash
python demo/gpt_visual_attention.py
```
Or, using `uv run`:
```bash
uv run python demo/gpt_visual_attention.py
```

## 📓 Notebook

- **`build-llm-from-scratch.ipynb`**: An educational notebook adapting concepts from the main repository for a step-by-step learning experience. You can run this using a Jupyter Notebook environment (e.g., by running `jupyter notebook` or `jupyter lab` from your activated environment).

Please refer to the individual scripts or the main book content for more detailed explanations of what each script does.
