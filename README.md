# PyCont Setup Guide (pyenv)

This project uses [pyenv] and [pyenv-virtualenv] to manage Python versions and virtual environments.

Tested with Python 3.13.7.

## Prerequisites

- Install pyenv and pyenv-virtualenv (see their docs for your OS):
  - https://github.com/pyenv/pyenv#installation
  - https://github.com/pyenv/pyenv-virtualenv#installation

## Setup Instructions (recommended)

1. Install Python 3.13.7 via pyenv:

  ```bash
  pyenv install 3.13.7
  ```

2. Create a local virtual environment for this repo and set it as the local version:

  ```bash
  pyenv virtualenv 3.13.7 PyCont_venv
  pyenv local PyCont_venv  # writes .python-version in this directory and points python interpreter to this virtual env
  python --version            # should show Python 3.13.7
  pyenv --versions           # should show PyCont_venv
  ```

1. Upgrade pip and build tools:

  ```bash
  python -m pip install --upgrade pip setuptools wheel
  ```

4. Install dependencies and the project (editable mode):

  ```bash
  pip install -r requirements.txt
  pip install -e .
  ```

  The `-e .` flag installs the project in editable mode, so code changes are reflected without reinstalling.

  To install without editable mode:

  ```bash
  pip install .
  ```

## Managing the environment (pyenv)

- pyenv will automatically activate the local environment when you `cd` into this directory (due to `.python-version`).
- If desired, to manually activate/deactivate in a shell session:

  ```bash
  pyenv activate PyCont_venv
  pyenv deactivate
  ```

## Uninstalling the package (from the env)

To remove the installed package (only from the current environment):

```bash
pip uninstall -y PyCont_lib_release
```
