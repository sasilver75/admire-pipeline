# ADMIRE Synthetic Data Pipeline

### Setup
- Install Pyenv: https://github.com/pyenv/pyenv
- Install the Python version specified in `.python-version` with Pyenv: `pyenv install 3.12.6`
- Create virtual environment: `python -m venv venv`
- Activate virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Edit any inputs as needed:
    - idiom lists in `data/*_idioms.txt`
    - image styles in `code/prompts/styles.py`
    - prompts in `code/prompts/prompts.py`
- Run the pipeline: `python code/run.py`

### Project Layout
- `code/`: Contains code for the pipeline.
    - `run.py`: Contains the main logic for the pipeline.
    - `prompts/`: Contains system and user prompts for compound sentence-use and image prompt generation, as well as collections of style modifiers injected into the prompts for diversity.
- `data/`: Contains the data for the ADMIRE task, as well as derived data.
    - `train/`: Contains images from the ADMIRE english training set.
    - `en_idioms.txt`: A text file of new-line separated English idiom compounds.
    - `README.md`: A README for the data directory containing information about idiom sources.
    - `subtask_a_train.tsv`: A tab-separated values file containing the ADMIRE english training set (besides images, which are in `train/`).
- `notebooks/`: Contains ipynb files for data exploration and minimal processing.
- `venv/`: Contains the virtual environment, git-ignored.
- `view_scripts/`: Contains scripts and templates for viewing the original training set.
- `.env`: Environment variables for the project (eg personal API keys), git-ignored.
- `.gitignore`: List of files to ignore for git.
- `.pre-commit-config.yaml`: Configuration for pre-commit hooks.
- `.python-version`: Used by Pyenv to manage the Python version.
- `pyproject.toml`: Configuration for project development tools (e.g. `black`, `isort`) that are used by pre-commit.
- `README.md`: This file.
- `requirements.txt`: List of dependencies for the project.