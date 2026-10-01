import json

import pytest

from tv_model.data import EOT
from tv_model.tokenizer import (
    EDGE_CASES,
    FORMAT,
    load_tokenizer,
    save_tokenizer,
    synthetic_samples,
    to_engine_json,
    tokenizer_cases,
    train_tokenizer,
)

STORIES = [
    "Once upon a time, there was a little girl named Lily. She liked to play outside.",
    "Tom had a red ball. He threw the ball to his dog, and the dog ran fast.",
    "The sun was hot. Lily and Tom sat under a big tree and ate an apple.",
] * 50


@pytest.fixture(scope="module")
def tokenizer():
    return train_tokenizer(STORIES, vocab_size=300)


def test_special_token_is_id_zero(tokenizer):
    assert tokenizer.token_to_id(EOT) == 0
    assert tokenizer.encode(f"a{EOT}b").ids.count(0) == 1


def test_engine_json_has_vocab_merges_and_specials(tokenizer):
    data = to_engine_json(tokenizer)
    assert data["type"] == FORMAT
    assert data["specialTokens"] == {EOT: 0}
    assert len(data["vocab"]) == tokenizer.get_vocab_size()
    assert all(isinstance(pair, list) and len(pair) == 2 for pair in data["merges"])
    assert all("".join(pair) in data["vocab"] for pair in data["merges"])


def test_save_and_load_encode_identically(tokenizer, tmp_path):
    path = tmp_path / "tokenizer.json"
    save_tokenizer(tokenizer, path)
    assert json.loads(path.read_text(encoding="utf-8"))["type"] == FORMAT
    reloaded = load_tokenizer(path)
    for text in [*EDGE_CASES, *STORIES[:3]]:
        assert reloaded.encode(text).ids == tokenizer.encode(text).ids, text


def test_load_rejects_other_formats(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"type": "something-else"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_tokenizer(path)


def test_decoding_is_lossless_for_edge_cases(tokenizer):
    for text in EDGE_CASES:
        assert tokenizer.decode(tokenizer.encode(text).ids, skip_special_tokens=False) == text


def test_cases_cover_edge_cases_then_samples_without_duplicates(tokenizer):
    cases = tokenizer_cases(tokenizer, ["Hello, world!", "A new sample."])
    texts = [case["text"] for case in cases]
    assert texts[: len(EDGE_CASES)] == EDGE_CASES
    assert texts[-1] == "A new sample."
    assert len(texts) == len(set(texts))
    assert all(case["ids"] == tokenizer.encode(case["text"]).ids for case in cases)


def test_synthetic_samples_are_seeded_strings_of_vocabulary_tokens(tokenizer):
    samples = synthetic_samples(tokenizer, 50, seed=0)
    assert samples == synthetic_samples(tokenizer, 50, seed=0)
    assert samples != synthetic_samples(tokenizer, 50, seed=1)
    assert len(samples) == 50
    assert all(len(text) >= 3 and "\ufffd" not in text and EOT not in text for text in samples)
