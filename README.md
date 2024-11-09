# ADMIRE Synthetic Data Pipeline

### Setup
- Install Pyenv: https://github.com/pyenv/pyenv
- Install the Python version specified in `.python-version` with Pyenv: `pyenv install 3.12.6`
- Create virtual environment: `python -m venv venv`
- Activate virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

### Project Layout
- `code/`: Contains code for the pipeline.
- `data/`: Contains the data for the ADMIRE task, as well as derived data.
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