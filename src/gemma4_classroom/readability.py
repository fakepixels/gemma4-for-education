from __future__ import annotations

import math
import re

WORD_RE = re.compile(r"[A-Za-z']+")
SENTENCE_RE = re.compile(r"[.!?]+")
VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


def split_words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def count_sentences(text: str) -> int:
    return max(1, len([part for part in SENTENCE_RE.split(text) if part.strip()]))


def count_syllables(word: str) -> int:
    word = word.lower().strip()
    if not word:
        return 0
    syllables = len(VOWEL_GROUP_RE.findall(word))
    if word.endswith("e") and syllables > 1:
        syllables -= 1
    return max(1, syllables)


def flesch_kincaid_grade(text: str) -> float:
    words = split_words(text)
    if not words:
      return 0.0
    sentences = count_sentences(text)
    syllables = sum(count_syllables(word) for word in words)
    words_per_sentence = len(words) / sentences
    syllables_per_word = syllables / len(words)
    return round((0.39 * words_per_sentence) + (11.8 * syllables_per_word) - 15.59, 2)


def avg_words_per_sentence(text: str) -> float:
    words = split_words(text)
    if not words:
        return 0.0
    return round(len(words) / count_sentences(text), 2)


def avg_word_length(text: str) -> float:
    words = split_words(text)
    if not words:
        return 0.0
    return round(sum(len(word) for word in words) / len(words), 2)


def lexical_diversity(text: str) -> float:
    words = [word.lower() for word in split_words(text)]
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 3)


def estimate_level_alignment(text: str, target_level: str) -> dict[str, float | bool]:
    grade = flesch_kincaid_grade(text)
    avg_sentence = avg_words_per_sentence(text)
    if target_level == "below":
        within_band = grade <= 4.5 and avg_sentence <= 12.5
    elif target_level == "on":
        within_band = 5.0 <= grade <= 7.5
    else:
        within_band = grade >= 7.0 and avg_sentence >= 11.5
    return {
        "readability_grade": grade,
        "avg_words_per_sentence": avg_sentence,
        "avg_word_length": avg_word_length(text),
        "lexical_diversity": lexical_diversity(text),
        "within_target_band": within_band,
    }
