import numpy as np

from tv_model.data import EOT
from tv_model.encode import encode_to_file, read_tokens
from tv_model.tokenizer import train_tokenizer

STORIES = ["Lily ran home.", "Tom ate a red apple.", "The dog was happy."] * 20


def test_writes_stories_separated_by_end_of_text(tmp_path):
    tokenizer = train_tokenizer(STORIES, vocab_size=280)
    path = tmp_path / "tokens.bin"
    count = encode_to_file(tokenizer, STORIES[:3], path, batch_size=2)
    tokens = read_tokens(path)
    assert tokens.dtype == np.uint16
    assert count == len(tokens)
    expected: list[int] = []
    for story in STORIES[:3]:
        expected.extend(tokenizer.encode(story).ids)
        expected.append(tokenizer.token_to_id(EOT))
    assert tokens.tolist() == expected
    text = tokenizer.decode(tokens.tolist(), skip_special_tokens=False)
    assert text == EOT.join(STORIES[:3]) + EOT


def test_file_is_little_endian_uint16(tmp_path):
    tokenizer = train_tokenizer(STORIES, vocab_size=280)
    path = tmp_path / "tokens.bin"
    encode_to_file(tokenizer, ["Lily ran home."], path)
    raw = path.read_bytes()
    first = tokenizer.encode("Lily ran home.").ids[0]
    assert raw[:2] == first.to_bytes(2, "little")
