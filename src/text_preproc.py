import re
from typing import List

def clean_text(text: str) -> str:
    """
    Lowercase, strip, normalize whitespace, remove non-printable chars.
    """
    if not text:
        return ""
    s = text.strip()
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.lower()
    return s

def tokenize(text: str) -> List[str]:
    """
    Simple whitespace + word tokenization.
    """
    text = clean_text(text)
    tokens = re.findall(r"\w+", text)
    return tokens

def text_to_summary_features(text: str) -> dict:
    """
    Returns lightweight features: char count, token count, first 20 tokens.
    """
    txt = clean_text(text)
    tokens = tokenize(txt)
    return {
        "char_count": len(txt),
        "token_count": len(tokens),
        "first_tokens": tokens[:20]
    }

# Test
if __name__ == "__main__":
    sample = "URGENT: Please reset your password NOW."
    print(clean_text(sample))
    print(tokenize(sample))
    print(text_to_summary_features(sample))
