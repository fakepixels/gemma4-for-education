from __future__ import annotations

import re

WORD_RE = re.compile(r"[A-Za-z']+")
SENTENCE_RE = re.compile(r"[.!?]+")
VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)
CLAUSE_MARKER_RE = re.compile(r"\b(because|although|while|whereas|which|including|however|therefore|since|when|if)\b", re.IGNORECASE)

SCIENCE_TERMS = {
    "atom",
    "atoms",
    "bacteria",
    "battery",
    "biodiversity",
    "cell",
    "cells",
    "cellular",
    "cellmembrane",
    "chloroplast",
    "chloroplasts",
    "circuit",
    "circuits",
    "community",
    "communities",
    "condense",
    "condenses",
    "condensation",
    "consumer",
    "consumers",
    "cytoplasm",
    "decomposer",
    "decomposers",
    "density",
    "diffusion",
    "dna",
    "ecosystem",
    "ecosystems",
    "electric",
    "electrical",
    "electron",
    "electrons",
    "element",
    "elements",
    "energy",
    "evaporate",
    "evaporates",
    "evaporation",
    "foodweb",
    "foodwebs",
    "fungi",
    "gas",
    "gases",
    "genetic",
    "genetics",
    "glucose",
    "gravity",
    "molecule",
    "molecules",
    "mitochondria",
    "nucleus",
    "nutrient",
    "nutrients",
    "organ",
    "organelle",
    "organelles",
    "organism",
    "organisms",
    "osmosis",
    "oxygen",
    "parallel",
    "particle",
    "particles",
    "photosynthesis",
    "plant",
    "plants",
    "population",
    "populations",
    "precipitation",
    "producer",
    "producers",
    "proton",
    "protons",
    "reaction",
    "reactions",
    "resistance",
    "respiration",
    "runoff",
    "sedimentary",
    "series",
    "solar",
    "species",
    "sunlight",
    "temperature",
    "thermal",
    "transpiration",
    "vapor",
    "voltage",
    "watercycle",
}

SUPPORT_MARKERS = (
    "called",
    "means",
    "which is",
    "this is",
    "such as",
    "for example",
    "also called",
)


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


def normalize_word(word: str) -> str:
    return re.sub(r"[^a-z]", "", word.lower())


def science_term_count(text: str) -> int:
    return sum(1 for word in split_words(text) if normalize_word(word) in SCIENCE_TERMS)


def complex_non_domain_ratio(text: str) -> float:
    words = split_words(text)
    if not words:
        return 0.0
    complex_words = [
        word for word in words
        if len(word) >= 8 and normalize_word(word) not in SCIENCE_TERMS
    ]
    return round(len(complex_words) / len(words), 3)


def support_marker_count(text: str) -> int:
    lowered = text.lower()
    return sum(lowered.count(marker) for marker in SUPPORT_MARKERS)


def clause_marker_density(text: str) -> float:
    words = split_words(text)
    if not words:
        return 0.0
    clause_markers = len(CLAUSE_MARKER_RE.findall(text))
    punctuation_markers = text.count(",") + text.count(";")
    return round((clause_markers + punctuation_markers) / len(words), 3)


def level_control_score(text: str, target_level: str) -> float:
    avg_sentence = avg_words_per_sentence(text)
    science_terms = science_term_count(text)
    complex_ratio = complex_non_domain_ratio(text)
    support_count = support_marker_count(text)
    clause_density = clause_marker_density(text)
    diversity = lexical_diversity(text)

    score = 1.0
    if target_level == "below":
        if avg_sentence > 11.0:
            score -= 0.3
        if complex_ratio > 0.12:
            score -= 0.35
        if clause_density > 0.11:
            score -= 0.2
        if science_terms >= 3 and support_count == 0:
            score -= 0.15
    elif target_level == "on":
        if avg_sentence < 7.0 or avg_sentence > 15.0:
            score -= 0.25
        if complex_ratio > 0.18:
            score -= 0.25
        if science_terms < 2:
            score -= 0.15
        if clause_density > 0.16:
            score -= 0.15
    else:
        if avg_sentence < 10.0:
            score -= 0.3
        if science_terms < 3:
            score -= 0.2
        if clause_density < 0.07:
            score -= 0.2
        if diversity < 0.45:
            score -= 0.1
    return round(max(0.0, min(1.0, score)), 3)


def estimate_level_alignment(text: str, target_level: str) -> dict[str, float | bool]:
    grade = flesch_kincaid_grade(text)
    avg_sentence = avg_words_per_sentence(text)
    science_terms = science_term_count(text)
    support_count = support_marker_count(text)
    clause_density = clause_marker_density(text)
    complex_ratio = complex_non_domain_ratio(text)
    control_score = level_control_score(text, target_level)
    if target_level == "below":
        within_band = control_score >= 0.65 and avg_sentence <= 12.5
    elif target_level == "on":
        within_band = control_score >= 0.65 and 7.0 <= avg_sentence <= 15.5
    else:
        within_band = control_score >= 0.65 and avg_sentence >= 9.5
    return {
        "readability_grade": grade,
        "avg_words_per_sentence": avg_sentence,
        "avg_word_length": avg_word_length(text),
        "lexical_diversity": lexical_diversity(text),
        "science_term_count": science_terms,
        "support_marker_count": support_count,
        "clause_marker_density": clause_density,
        "complex_non_domain_ratio": complex_ratio,
        "level_control_score": control_score,
        "within_target_band": within_band,
    }
