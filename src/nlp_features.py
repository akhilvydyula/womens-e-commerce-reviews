"""Handcrafted NLP features (lexicon + distributional stats) for review text."""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

_WORD = re.compile(r"[a-zA-Z']+")
_SENT_SPLIT = re.compile(r"[.!?]+")

POSITIVE = frozenset(
    "love great perfect amazing excellent wonderful recommend best beautiful "
    "happy satisfied favorite awesome fantastic comfortable flattering gorgeous "
    "stunning brilliant superb pleased delighted thrilled outstanding".split()
)
NEGATIVE = frozenset(
    "hate terrible awful disappointed bad poor worst never return cheap ugly "
    "uncomfortable wrong misleading horrible defective ripped stained faded "
    "scratchy itchy ill-fitting overpriced underwhelming regret waste".split()
)
INTENSIFIERS = frozenset(
    "very really extremely incredibly absolutely totally utterly highly so too "
    "especially exceptionally particularly".split()
)
NEGATIONS = frozenset(
    "not never no nor neither nowhere nothing nobody hardly barely scarcely "
    "without isn't aren't wasn't weren't don't doesn't didn't won't wouldn't "
    "can't couldn't shouldn't".split()
)
FIT_WORDS = frozenset(
    "fit fits fitted fitting size sized sizing tight loose snug oversized "
    "petite plus true-to-size tts runs small runs large boxy tailored".split()
)
QUALITY_WORDS = frozenset(
    "quality material fabric construction stitch seam durable durability "
    "craftsmanship premium cheap flimsy sheer transparent lining".split()
)
PRICE_WORDS = frozenset(
    "price priced pricing expensive costly affordable value worth money "
    "overpriced underpriced sale discount bargain steal".split()
)
SHIPPING_WORDS = frozenset(
    "shipping shipped delivery arrived package packaging transit delay "
    "late early fast slow courier".split()
)
RETURN_WORDS = frozenset(
    "return returned returning refund exchange send back".split()
)
RECOMMEND_PHRASES = (
    "would recommend",
    "highly recommend",
    "recommend to",
    "recommend this",
    "tell friends",
    "buy again",
    "purchase again",
)

NLP_FEATURE_NAMES = [
    "nlp_char_count",
    "nlp_word_count",
    "nlp_sentence_count",
    "nlp_avg_word_len",
    "nlp_unique_word_ratio",
    "nlp_exclamation_count",
    "nlp_question_count",
    "nlp_uppercase_ratio",
    "nlp_digit_count",
    "nlp_sentiment_pos",
    "nlp_sentiment_neg",
    "nlp_sentiment_net",
    "nlp_intensifier_count",
    "nlp_negation_count",
    "nlp_first_person_count",
    "nlp_fit_mentions",
    "nlp_quality_mentions",
    "nlp_price_mentions",
    "nlp_shipping_mentions",
    "nlp_return_mentions",
    "nlp_recommend_signal",
    "nlp_avg_sentence_len",
    "nlp_repetition_ratio",
    "nlp_emoji_count",
    "nlp_hashtag_count",
]


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _count_lexicon(tokens: Iterable[str], lexicon: frozenset[str]) -> int:
    return sum(1 for t in tokens if t in lexicon)


def _phrase_hits(text: str, phrases: tuple[str, ...]) -> int:
    low = text.lower()
    return sum(1 for p in phrases if p in low)


def extract_nlp_features(title: pd.Series, review: pd.Series) -> pd.DataFrame:
    combined = (title.fillna("") + " " + review.fillna("")).astype(str)
    rows = []
    for text in combined:
        tokens = _tokenize(text)
        n_words = max(len(tokens), 1)
        sentences = [s for s in _SENT_SPLIT.split(text) if s.strip()]
        n_sent = max(len(sentences), 1)
        unique_ratio = len(set(tokens)) / n_words
        upper_chars = sum(1 for c in text if c.isupper())
        char_n = max(len(text), 1)

        rows.append(
            {
                "nlp_char_count": len(text),
                "nlp_word_count": len(tokens),
                "nlp_sentence_count": n_sent,
                "nlp_avg_word_len": float(np.mean([len(t) for t in tokens]) if tokens else 0),
                "nlp_unique_word_ratio": unique_ratio,
                "nlp_exclamation_count": text.count("!"),
                "nlp_question_count": text.count("?"),
                "nlp_uppercase_ratio": upper_chars / char_n,
                "nlp_digit_count": sum(c.isdigit() for c in text),
                "nlp_sentiment_pos": _count_lexicon(tokens, POSITIVE),
                "nlp_sentiment_neg": _count_lexicon(tokens, NEGATIVE),
                "nlp_sentiment_net": _count_lexicon(tokens, POSITIVE)
                - _count_lexicon(tokens, NEGATIVE),
                "nlp_intensifier_count": _count_lexicon(tokens, INTENSIFIERS),
                "nlp_negation_count": _count_lexicon(tokens, NEGATIONS),
                "nlp_first_person_count": sum(
                    1 for t in tokens if t in {"i", "me", "my", "mine", "myself"}
                ),
                "nlp_fit_mentions": _count_lexicon(tokens, FIT_WORDS),
                "nlp_quality_mentions": _count_lexicon(tokens, QUALITY_WORDS),
                "nlp_price_mentions": _count_lexicon(tokens, PRICE_WORDS),
                "nlp_shipping_mentions": _count_lexicon(tokens, SHIPPING_WORDS),
                "nlp_return_mentions": _count_lexicon(tokens, RETURN_WORDS),
                "nlp_recommend_signal": _phrase_hits(text, RECOMMEND_PHRASES),
                "nlp_avg_sentence_len": len(tokens) / n_sent,
                "nlp_repetition_ratio": 1.0 - unique_ratio,
                "nlp_emoji_count": len(
                    [c for c in text if ord(c) > 0x2600 and ord(c) < 0x27BF]
                ),
                "nlp_hashtag_count": text.count("#"),
            }
        )
    return pd.DataFrame(rows, columns=NLP_FEATURE_NAMES)
