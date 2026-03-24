import re
import unicodedata


SUBSTITUTIONS = {
    '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's',
    '7': 't', '@': 'a', '$': 's', '!': 'i', '+': 't',
}

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

def decode_leetspeak(text: str) -> str:
    return ''.join(SUBSTITUTIONS.get(c, c) for c in text)

def remove_zero_width_chars(text: str) -> str:
    return re.sub(r'[\u200b\u200c\u200d\ufeff\u00ad]', '', text)

def sanitize_input(text: str) -> dict:
    original = text
    text = remove_zero_width_chars(text)
    text = normalize_unicode(text)
    text_decoded = decode_leetspeak(text)
    evasion_detected = text_decoded != text.lower()

    return {
        'cleaned_text': text_decoded,
        'original_text': original,
        'evasion_detected': evasion_detected,
        'unicode_normalized': original != normalize_unicode(original),
    }