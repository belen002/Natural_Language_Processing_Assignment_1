
import sqlite3
from pathlib import Path

import streamlit as st

# Project paths
PROJECT_DIR = Path(r"D:\GPAC")
DB_PATH = PROJECT_DIR / "ngram_counts_local.sqlite"

st.set_page_config(
    page_title="Amharic Predictive Text",
    page_icon="📝",
    layout="centered",
)

st.title("Amharic Predictive Text")
st.write(
    "Enter an Amharic sentence and get suggestions "
    "for the next word."
)

st.caption("Prototype: unigram and bigram model")

@st.cache_data
def get_suggestions(previous_word, limit=5):
    """Return the most frequent words following the given word."""
    if not DB_PATH.exists():
        return []

    with sqlite3.connect(
        f"file:{DB_PATH.as_posix()}?mode=ro", uri=True
    ) as conn:
        rows = conn.execute(
            """
            SELECT w2, count
            FROM bigrams
            WHERE w1 = ?
            ORDER BY count DESC
            LIMIT ?
            """,
            (previous_word, limit),
        ).fetchall()

        total = conn.execute(
            """
            SELECT SUM(count)
            FROM bigrams
            WHERE w1 = ?
            """,
            (previous_word,),
        ).fetchone()[0]

    if not total:
        return []

    return [
        {
            "word": word,
            "count": count,
            "probability": count / total,
        }
        for word, count in rows
    ]


sentence = st.text_area(
    "Enter a sentence in Amharic",
    placeholder="ለምሳሌ፦ ኢትዮጵያ በአፍሪካ",
    height=120,
)

if st.button("Suggest next words", type="primary"):
    words = sentence.strip().split()

    if not words:
        st.warning("Please enter a sentence first.")
    elif not DB_PATH.exists():
        st.error(f"Database not found: {DB_PATH}")
    else:
        previous_word = words[-1]
        suggestions = get_suggestions(previous_word)

        if suggestions:
            st.subheader("Top 5 next-word suggestions")

            for item in suggestions:
                probability = item["probability"]
                st.write(
                    f"**{item['word']}** — "
                    f"{probability:.2%}"
                )

                if st.button(
                    f"Choose {item['word']}",
                    key=f"choose_{item['word']}",
                ):
                    updated = (
                        sentence.rstrip()
                        + " "
                        + item["word"]
                    )
                    st.session_state["continued_sentence"] = updated

        else:
            st.info(
                f"No bigram suggestions found for "
                f"'{previous_word}'. Try another word."
            )

if "continued_sentence" in st.session_state:
    st.subheader("Continued sentence")
    st.text_area(
        "Select and continue",
        key="continued_sentence",
        height=100,
    )
