import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.pii_detector import detect_pii, scrub_pii

def test_detect_email():
    result = detect_pii('contact me at test@example.com please')
    assert 'email' in result
    assert result['email'] == 1

def test_detect_phone():
    result = detect_pii('call me at 123-456-7890')
    assert 'phone' in result
    assert result['phone'] == 1

def test_detect_ssn():
    result = detect_pii('my ssn is 123-45-6789')
    assert 'ssn' in result
    assert result['ssn'] == 1

def test_detect_ip():
    result = detect_pii('server at 192.168.1.1')
    assert 'ip_address' in result
    assert result['ip_address'] == 1

def test_detect_multiple():
    result = detect_pii('email test@example.com and phone 123-456-7890')
    assert 'email' in result
    assert 'phone' in result

def test_detect_no_pii():
    result = detect_pii('this is a normal comment with no pii')
    assert result == {}

def test_scrub_email():
    result = scrub_pii('contact me at test@example.com')
    assert '[EMAIL_REDACTED]' in result
    assert 'test@example.com' not in result

def test_scrub_phone():
    result = scrub_pii('call me at 123-456-7890')
    assert '[PHONE_REDACTED]' in result
    assert '123-456-7890' not in result

def test_scrub_multiple():
    result = scrub_pii('email test@example.com phone 123-456-7890')
    assert '[EMAIL_REDACTED]' in result
    assert '[PHONE_REDACTED]' in result