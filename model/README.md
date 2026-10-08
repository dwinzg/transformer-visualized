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

The same steps train the Llama-style model in `models/llama-tiny` with `--preset llama-tiny`. Its model card lists the commands.


Text fed to the tiny model needs the same cleaning the training data went through; see the
"Input text" section in `models/tiny/README.md`.

This downloads about 2.2 GB of TinyStories into `model/data/`, which reaches about 3.1 GB once
the tokenized train and validation files are written, and trains for about 47 minutes on an
Apple M4. Without an MPS or CUDA device, training falls back to CPU and is much slower.

Retraining is not bit-reproducible on MPS, so a retrain gives a model that is comparable but
not identical to the committed one. After retraining, regenerate the model card, `models/tiny/`
and the engine fixtures, since they all describe this exact run. Running `prepare` also
overwrites the committed `models/tiny/tokenizer.json` and
`packages/engine/test/fixtures/tokenizer/cases.json`; they come out byte-identical when run
against the same revision of the dataset, but check before committing.

```sh
.venv/bin/python -m tv_model.prepare --data-dir data --tokenizer-out ../models/tiny/tokenizer.json --cases-out ../packages/engine/test/fixtures/tokenizer/cases.json
.venv/bin/python -m tv_model.train --preset tiny --data-dir data --out-dir runs/tiny
.venv/bin/python -m tv_model.export --run runs/tiny --out ../models/tiny
.venv/bin/python -m tv_model.fixtures tiny --model ../models/tiny/model.safetensors --tokenizer ../models/tiny/tokenizer.json --out ../packages/engine/test/fixtures/tiny
```

## Snapshots for the training page

The site's Watch it learn page shows a second run of the same recipe, saved at 17 steps. To make it again, train with `--snapshot-steps`, then turn the snapshots into the small JSON the page reads. The snapshots stay in `runs/`, and only the JSON is committed.

```sh
.venv/bin/python -m tv_model.train --preset tiny --data-dir data --out-dir runs/snapshots --snapshot-steps 0,25,50,100,150,200,300,400,500,750,1000,1500,2000,3000,5000,10000,25000
.venv/bin/python -m tv_model.training_views --snapshots runs/snapshots/snapshots --tokenizer ../models/tiny/tokenizer.json --out ../models/tiny/training.json
```

To generate the samples on the model card, run `tv_model.sample` with the shipped weights, one
of the card's prompts and one of its seeds:

```sh
.venv/bin/python -m tv_model.sample --weights ../models/tiny/model.safetensors --tokenizer ../models/tiny/tokenizer.json --prompt "Once upon a time" --tokens 80 --temperature 0.8 --top-k 40 --seed 0
```
