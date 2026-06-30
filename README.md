# Airoh Template: Reproducible Pipelines Made Simple

_why don't you have a cup of relaxing jasmine tea?_

This repository is a template for structuring a reproducible data analysis. Built on the [`invoke`](https://www.pyinvoke.org/) task runner, it lets you go from clean clone to output figures with just a few commands.

The logic is powered by [`airoh`](https://pypi.org/project/airoh/), a lightweight, pip-installable Python package of reusable `invoke` tasks. This repository runs a small demo analysis to show how the template works. It should be easy to adapt to a variety of projects.

**This template is designed to be used with [Claude Code](https://claude.ai/code).** Claude reads the project's `CLAUDE.md` at the start of every session and knows the pipeline conventions — task naming, idempotency, smoke tests — out of the box. To initialize a new project from this template, open Claude Code and run `/init-airoh-project`. The skill will walk you through project setup, fetch/run/clean task implementation, and a smoke test end-to-end.

⚠️ **Status**: This template is in its early days. Expect rapid iteration and changes.

---

## ✨ TL;DR:

This repository is a [GitHub template](https://github.com/airoh-pipeline/airoh-template/generate). Click **"Use this template"** to create your own analysis project.
```bash
uv sync
uv run invoke fetch
uv run invoke run
```
Voilà — from clone to full reproduction.

---

## 🚀 Quick Start

### **Step 1**: Install dependencies

Using `uv` (recommended):
```bash
uv sync
```
This creates a `.venv` and installs all dependencies from `pyproject.toml`.

Using `pip` (e.g. in a virtual environment):
```bash
pip install -r requirements.txt
```

Using `conda`:
```bash
conda env create -n airoh_env -f environment.yml
conda activate airoh_env
```

---

### **Step 2**: Fetch the source data

```bash
invoke fetch
```

Downloads the configured file(s) listed under `files:` in `invoke.yaml`.

---

### **Step 3**: Run the full pipeline

```bash
invoke run
```

Runs the full analysis pipeline in order. Steps that have already produced output are skipped automatically — only missing outputs are recomputed. To force a full rerun from scratch:

```bash
invoke clean
invoke run
```

---

### **Step 4**: Clean outputs

```bash
invoke clean          # remove all outputs
invoke clean-{name}   # remove outputs of one specific step
```

---

## 🧠 Design principles

Airoh projects follow a few conventions that keep analyses fast, reproducible, and easy to pick up:

- **Analysis in code, visualization in notebooks.** Heavy computation lives in `analysis/` Python modules and is run by `invoke` tasks. Notebooks only read results and produce figures — so they stay fast.
- **Idempotent steps.** Each `run-{name}` task checks whether its outputs already exist and skips if they do. You can call `invoke run` repeatedly while working on a later step without re-running earlier ones.
- **Mirrored clean tasks.** Every `run-{name}` has a matching `clean-{name}` that removes only its outputs. The top-level `clean` calls them all.
- **Smoke test.** Tasks support a `--smoke` flag for a fast minimal run to verify the pipeline end-to-end.

---

## 🧰 Task Overview

| Task             | Description                                              |
| ---------------- | -------------------------------------------------------- |
| `fetch`          | Downloads source data configured in `invoke.yaml`        |
| `run`            | Runs the full pipeline (all `run-{name}` steps in order) |
| `run-{name}`     | Runs one analysis step; skips if outputs already exist   |
| `run-notebooks`  | Executes notebooks and saves figures to `output_data/`   |
| `clean`          | Removes all generated outputs                            |
| `clean-{name}`   | Removes outputs of one specific step                     |

Use `invoke --list` or `invoke --help <task>` for descriptions and usage.

---

## 📁 Folder Structure

| Folder / File  | Description                              |
| -------------- | ---------------------------------------- |
| `analysis/`    | Pure Python analysis logic, called by invoke tasks |
| `notebooks/`   | Jupyter notebooks for visualization (one per figure) |
| `source_data/` | Raw source datasets — see [`source_data/CONTENT.md`](source_data/CONTENT.md) |
| `output_data/` | Generated results and figures — see [`output_data/CONTENT.md`](output_data/CONTENT.md) |
| `tasks.py`     | Project-specific invoke tasks            |
| `invoke.yaml`  | Config: paths, data sources, parameters  |

---

## 🧭 Tips

* Use `invoke --complete` for tab-completion support
* Configure paths and data sources in `invoke.yaml`
* To use this template for a new project, start from [`airoh-template`](https://github.com/airoh-pipeline/airoh-template) and customize `tasks.py` + `invoke.yaml`

---

## 🔁 Want to contribute?

Submit an issue or PR on [`airoh`](https://github.com/SIMEXP/airoh).

---

## Philosophy

Inspired by Uncle Iroh from *Avatar: The Last Airbender*, `airoh` aims to bring simplicity, reusability, and clarity to research infrastructure — one well-structured task at a time.

**Core principles:**

- **Reproducibility first.** A pipeline is only useful if someone else — or future you — can run it from scratch and get the same result. Every step is scripted, every dependency declared.
- **Simple by default, extensible by need.** Three tasks (`fetch`, `run`, `clean`) cover most projects. Add complexity only when the analysis demands it.
- **Code for analysis, notebooks for figures.** Heavy computation belongs in `analysis/` Python modules. Notebooks are for reading results and producing plots — they should be fast and focused.
- **Idempotent steps.** Re-running `invoke run` never wastes time. Each step checks whether its outputs exist and skips if they do.
- **AI-native.** This template is built to be initialized and extended with Claude Code. The `CLAUDE.md` file gives Claude the context it needs to help with the pipeline without needing to re-explain conventions every session.

---

### Uncle Airoh

When working in this project, Claude Code responds as **Uncle Airoh**: patient, warm, and wise. Errors are explained gently, tradeoffs are framed as learning opportunities, and a calming cup of jasmine tea is always on offer when things get heated.
