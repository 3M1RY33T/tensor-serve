from utils import clean_text


def test_clean_text():
    html = "<p>Hello <b>World</b>\n<br>   This is <i>test</i></p>"

    assert clean_text(html) == "Hello World This is test"
