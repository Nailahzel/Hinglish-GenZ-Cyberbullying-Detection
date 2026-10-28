import re
import emoji
from text_unidecode import unidecode

URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
USER = re.compile(r"@\w+")
SPACE = re.compile(r"\s+")

def normalize_text(s):
    if not isinstance(s, str):
        return ""
    s = URL.sub(" <url> ", s)
    s = USER.sub(" <user> ", s)
    s = unidecode(s)
    s = re.sub(r"(\w)\1{2,}", r"\1\1", s)
    s = SPACE.sub(" ", s)
    return s.strip()

def keep_emojis(s):
    return "".join(ch for ch in s if ch in emoji.EMOJI_DATA) or s
