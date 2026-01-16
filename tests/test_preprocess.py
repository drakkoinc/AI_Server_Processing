from app.preprocess import html_to_text, extract_links, extract_money_expressions, extract_time_expressions

def test_html_to_text():
    html = "<html><body><p>Hello</p><p>World</p></body></html>"
    text = html_to_text(html)
    assert "Hello" in text
    assert "World" in text

def test_extract_links():
    text = "See https://example.com and http://test.com/path"
    links = extract_links(text)
    assert "https://example.com" in links

def test_extract_money():
    text = "Invoice total is $1,234.56 due soon"
    money = extract_money_expressions(text)
    assert "$1,234.56" in money

def test_extract_time():
    text = "Please send by Friday EOD, thanks."
    times = extract_time_expressions(text)
    assert any("by friday" in t.lower() for t in [x.lower() for x in times])
