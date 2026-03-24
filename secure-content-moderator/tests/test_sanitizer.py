import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.input_sanitizer import normalize_unicode, decode_leetspeak, remove_zero_width_chars, sanitize_input

def test_normalize_unicode():
    assert normalize_unicode('аsshole') == 'sshole'  

def test_decode_leetspeak():
    assert decode_leetspeak('1d10t') == 'idiot'
    assert decode_leetspeak('@$$hole') == 'asshole'
    assert decode_leetspeak('h3ll0') == 'hello'
    assert decode_leetspeak('normal text') == 'normal text'

def test_remove_zero_width_chars():
    text_with_zwsp = 'i\u200bd\u200bi\u200bo\u200bt'
    assert remove_zero_width_chars(text_with_zwsp) == 'idiot'

def test_sanitize_input_detects_evasion():
    result = sanitize_input('1d10t')
    assert result['evasion_detected'] == True
    assert result['cleaned_text'] == 'idiot'  

def test_sanitize_input_clean_text():
    result = sanitize_input('hello world')
    assert result['evasion_detected'] == False
    assert result['cleaned_text'] == 'hello world'

def test_sanitize_input_returns_original():
    result = sanitize_input('1d10t')
    assert result['original_text'] == '1d10t'