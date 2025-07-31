# PyCont Setup Guide

This project uses a Python virtual environment (`venv`) to manage dependencies cleanly and avoid system conflicts.

## Setup Instructions

1. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies and the project in editable mode:**

   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

   The `-e .` flag installs the project in *editable mode*, so any code changes are reflected without reinstalling.

   You can also install the project without editable mode using:

   ```bash
   pip install .
   ```

## Uninstalling the Package (from venv)

To remove the installed package (only from the virtual environment):

```bash
pip uninstall -y PyCont_lib
```

## Notes

* You must **activate the virtual environment** every time before working:

  ```bash
  source venv/bin/activate
  ```

* To deactivate:

  ```bash
  deactivate
  ```

* The `venv/` folder should be excluded from version control. Add this to your `.gitignore`:

  ```
  venv/
  ```
