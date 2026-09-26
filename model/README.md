# Model

Python code behind the model on the site. It defines the GPT-2 style network in PyTorch and generates the golden fixtures that the TypeScript engine is tested against.

## Setup

You need Python 3.12.

```sh
cd model
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Tests and checks

```sh
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

## Engine fixtures

The engine's parity tests compare every intermediate value with this model. Regenerate the fixtures after changing the model.

```sh
.venv/bin/python -m tv_model.fixtures micro --out ../packages/engine/test/fixtures/micro
```
