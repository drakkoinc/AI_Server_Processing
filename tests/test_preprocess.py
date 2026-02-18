from app.preprocess import html_to_text, extract_links, extract_money_expressions, extract_time_expressions, extract_unsubscribe_signals

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


def test_extract_unsubscribe_signals_positive():
    assert extract_unsubscribe_signals("Click here to unsubscribe from this list") is True
    assert extract_unsubscribe_signals("subscribe to our newsletter") is True
    assert extract_unsubscribe_signals("change your preferences here") is True
    assert extract_unsubscribe_signals("Change preferences") is True
    assert extract_unsubscribe_signals("edit preferences for your account") is True
    assert extract_unsubscribe_signals("update subscription settings") is True
    assert extract_unsubscribe_signals("update preferences") is True
    assert extract_unsubscribe_signals("opt out of future emails") is True
    assert extract_unsubscribe_signals("opt-out") is True
    assert extract_unsubscribe_signals("manage preferences") is True
    assert extract_unsubscribe_signals("mailing list") is True


def test_extract_unsubscribe_signals_negative():
    assert extract_unsubscribe_signals("Hi, can you confirm the meeting time?") is False
    assert extract_unsubscribe_signals("Please review the attached document.") is False
    assert extract_unsubscribe_signals("Invoice #1234 due by Friday") is False
    assert extract_unsubscribe_signals("") is False
