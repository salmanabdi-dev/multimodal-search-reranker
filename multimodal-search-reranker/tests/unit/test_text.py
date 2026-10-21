from mmsearch.ingestion.text import clean_text, passage_text


def test_clean_text_collapses_whitespace_and_keeps_paragraphs():
    raw = "Hello\u00a0  world\t!\r\n\r\n\r\n\r\nSecond\x07 para  "
    assert clean_text(raw) == "Hello world !\n\nSecond para"


def test_clean_text_nfc():
    assert clean_text("cafe\u0301") == "caf\u00e9"


def test_passage_text():
    assert passage_text(None, "body") == "body"
    assert passage_text("Title", "body") == "Title. body"
    assert passage_text("Why?", "body") == "Why? body"
