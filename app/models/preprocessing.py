import string

def simple_preprocessor(text: str) -> str:
    return text

def no_punctuation_preprocessor(text: str) -> str:
    blacklist_chars = string.punctuation
    translation_table = str.maketrans("", "", blacklist_chars)
    cleaned_text = text.translate(translation_table)
    return cleaned_text

preprocessors = {
    "simple": simple_preprocessor,
    "no_punct": no_punctuation_preprocessor
}