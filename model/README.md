# Model

Python code behind the model on the site. It defines the GPT-2 style network in PyTorch and generates the golden fixtures that the TypeScript engine is tested against.

## Setup

You need Python 3.12.

```sh
cd model
python3.12 -m venv .venv
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

## Training the tiny model

Text fed to the tiny model needs the same cleaning the training data went through; see the
"Input text" section in `models/tiny/README.md`.

This downloads about 2.2 GB of TinyStories into `model/data/` and trains for about 47 minutes on an Apple M4.

```sh
.venv/bin/python -m tv_model.prepare --data-dir data --tokenizer-out ../models/tiny/tokenizer.json --cases-out ../packages/engine/test/fixtures/tokenizer/cases.json
.venv/bin/python -m tv_model.train --preset tiny --data-dir data --out-dir runs/tiny
.venv/bin/python -m tv_model.export --run runs/tiny --out ../models/tiny
.venv/bin/python -m tv_model.fixtures tiny --model ../models/tiny/model.safetensors --tokenizer ../models/tiny/tokenizer.json --out ../packages/engine/test/fixtures/tiny
```
