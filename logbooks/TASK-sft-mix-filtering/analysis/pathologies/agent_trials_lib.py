import unicodedata
SCRIPTS = ("CJK", "HIRAGANA", "KATAKANA", "HANGUL", "CYRILLIC", "GREEK", "ARABIC", "HEBREW", "THAI", "DEVANAGARI",
           "TAMIL", "TELUGU", "BENGALI", "GEORGIAN", "ARMENIAN", "MONGOLIAN", "ETHIOPIC", "KHMER", "MYANMAR", "TIBETAN")
def scripts(s):
    seen = set()
    for ch in s:
        if ord(ch) < 0x370: continue
        name = unicodedata.name(ch, "")
        for sc in SCRIPTS:
            if name.startswith(sc): seen.add(sc); break
    return seen
def token_soup(s):
    return len(scripts(s)) >= 4 or s.count("�") >= 2
