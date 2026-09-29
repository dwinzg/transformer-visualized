from tv_model.data import EOT, clean_stories, clean_story, iter_stories, normalize


def test_normalize_maps_typography_to_ascii():
    text = "\u201cHi,\u201d she said\u2026 it\u2019s \u2013 fine\u2014ok\u00a0now "
    assert normalize(text) == '"Hi," she said... it\'s - fine-ok now'


def test_iter_stories_splits_on_end_of_text_lines(tmp_path):
    path = tmp_path / "stories.txt"
    path.write_text(
        f"One day.\nThe end.\n{EOT}\n\nSecond story.\n{EOT}\nTrailing story\n", encoding="utf-8"
    )
    assert list(iter_stories(path)) == ["One day.\nThe end.", "Second story.", "Trailing story"]


def test_iter_stories_skips_empty_stories(tmp_path):
    path = tmp_path / "stories.txt"
    path.write_text(f"{EOT}\n\n{EOT}\nOnly one.\n", encoding="utf-8")
    assert list(iter_stories(path)) == ["Only one."]


def test_clean_story_returns_none_when_empty_or_still_non_ascii():
    assert clean_story(" It\u2019s fine. ") == "It's fine."
    assert clean_story("Caf\u00e9 time.") is None
    assert clean_story("   ") is None


def test_clean_stories_keeps_plain_ascii_after_normalizing():
    stories = ["It\u2019s fine.", "Caf\u00e9 time.", "Plain.", "   ", "Bad \x92 byte."]
    assert list(clean_stories(stories)) == ["It's fine.", "Plain."]
