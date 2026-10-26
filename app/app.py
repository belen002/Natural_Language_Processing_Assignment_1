
import re
import sqlite3
from pathlib import Path

import streamlit as st
from huggingface_hub import hf_hub_download

# --------------------------------------------------
# DATABASE CONFIGURATION
# --------------------------------------------------

# Original local database configuration — kept for reference.
# Uncomment these lines if you want to run the app locally
# using the database on your computer.
#
# PROJECT_DIR = Path(r"D:\GPAC")
# DB_PATH = PROJECT_DIR / "ngram_counts_combined.sqlite"

# Active configuration: download the existing database
# from the public Hugging Face dataset repository.
HF_REPO_ID = "BeamlakF/amharic-ngram-database"
HF_FILENAME = "ngram_counts_combined.sqlite"


@st.cache_resource(show_spinner="Preparing the n-gram database...")
def get_db_path() -> Path:
    """Download the database through Hugging Face and return its cached path."""
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        repo_type="dataset",
    )
    return Path(path)


st.set_page_config(
    page_title="Amharic Predictive Text",
    page_icon="📝",
    layout="centered",
)

st.title("Amharic Predictive Text")
st.write("Enter an Amharic sentence and get five next-word suggestions.")
st.caption("Uses unigram, bigram, trigram, and four-gram frequency models.")

TOKEN_RE = re.compile(
    r"""[^\s፡።፣፤፥፦፧፨?!,.;:()\[\]{}<>"'/%—–-]+"""
)


def tokenize(text: str) -> list[str]:
    """Split text into word-like tokens, excluding common punctuation."""
    return TOKEN_RE.findall(text)


def open_db():
    """Open the downloaded SQLite database in read-only mode."""
    db_path = get_db_path()
    uri = db_path.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=10)


def append_suggestion(word: str) -> None:
    """Append a selected word before Streamlit rebuilds the widgets."""
    current = st.session_state.get("sentence_input", "")
    if current and not current[-1].isspace():
        current += " "
    st.session_state["sentence_input"] = current + word
    st.session_state["prediction_results"] = []
    st.session_state["prediction_model"] = ""


def reset_app() -> None:
    """Clear app state before the widgets are recreated."""
    st.session_state["sentence_input"] = ""
    st.session_state["prediction_results"] = []
    st.session_state["prediction_model"] = ""


@st.cache_data(show_spinner=False)
def predict_next_words(context_words: tuple[str, ...], limit: int = 5):
    """Use longest available context: four-gram, trigram, bigram, then unigram."""
    with open_db() as conn:
        conn.execute("PRAGMA query_only = ON")

        if len(context_words) >= 3:
            w1, w2, w3 = context_words[-3:]
            rows = conn.execute(
                """SELECT w4, count FROM fourgrams
                   WHERE w1=? AND w2=? AND w3=?
                   ORDER BY count DESC LIMIT ?""",
                (w1, w2, w3, limit),
            ).fetchall()

            if rows:
                row = conn.execute(
                    """SELECT outgoing FROM fourgram_contexts
                       WHERE w1=? AND w2=? AND w3=? LIMIT 1""",
                    (w1, w2, w3),
                ).fetchone()

                denominator = (
                    row[0] if row and row[0]
                    else sum(r[1] for r in rows)
                )

                return [
                    {
                        "word": word,
                        "count": count,
                        "probability": count / denominator,
                    }
                    for word, count in rows
                ], "four-gram"

        if len(context_words) >= 2:
            w1, w2 = context_words[-2:]
            rows = conn.execute(
                """SELECT w3, count FROM trigrams
                   WHERE w1=? AND w2=?
                   ORDER BY count DESC LIMIT ?""",
                (w1, w2, limit),
            ).fetchall()

            if rows:
                row = conn.execute(
                    """SELECT outgoing FROM trigram_contexts
                       WHERE w1=? AND w2=? LIMIT 1""",
                    (w1, w2),
                ).fetchone()

                denominator = (
                    row[0] if row and row[0]
                    else sum(r[1] for r in rows)
                )

                return [
                    {
                        "word": word,
                        "count": count,
                        "probability": count / denominator,
                    }
                    for word, count in rows
                ], "trigram"

        if context_words:
            previous_word = context_words[-1]
            rows = conn.execute(
                """SELECT w2, count FROM bigrams
                   WHERE w1=?
                   ORDER BY count DESC LIMIT ?""",
                (previous_word, limit),
            ).fetchall()

            if rows:
                total = conn.execute(
                    "SELECT SUM(count) FROM bigrams WHERE w1=?",
                    (previous_word,),
                ).fetchone()[0]

                denominator = total or sum(r[1] for r in rows)

                return [
                    {
                        "word": word,
                        "count": count,
                        "probability": count / denominator,
                    }
                    for word, count in rows
                ], "bigram"

        rows = conn.execute(
            """SELECT token, count FROM unigrams
               ORDER BY count DESC LIMIT ?""",
            (limit,),
        ).fetchall()

        total = conn.execute(
            "SELECT SUM(count) FROM unigrams"
        ).fetchone()[0]

        if not rows or not total:
            return [], "none"

        return [
            {
                "word": word,
                "count": count,
                "probability": count / total,
            }
            for word, count in rows
        ], "unigram"


if "sentence_input" not in st.session_state:
    st.session_state["sentence_input"] = ""

if "prediction_results" not in st.session_state:
    st.session_state["prediction_results"] = []

if "prediction_model" not in st.session_state:
    st.session_state["prediction_model"] = ""


st.text_area(
    "Enter a sentence in Amharic",
    key="sentence_input",
    placeholder="ለምሳሌ፦ ኢትዮጵያ በአፍሪካ",
    height=120,
)

col1, col2 = st.columns(2)

with col1:
    suggest_clicked = st.button(
        "Suggest next words",
        type="primary",
        use_container_width=True,
    )

with col2:
    st.button(
        "Reset",
        on_click=reset_app,
        use_container_width=True,
    )


if suggest_clicked:
    words = tokenize(st.session_state["sentence_input"].strip())

    if not words:
        st.warning("Please enter at least one word first.")
        st.session_state["prediction_results"] = []
        st.session_state["prediction_model"] = ""
    else:
        try:
            # Download on first use; reuse the Hugging Face cache afterwards.
            with st.spinner("Loading the n-gram database..."):
                db_path = get_db_path()

            if not db_path.exists() or db_path.stat().st_size == 0:
                st.error("The downloaded database is missing or empty.")
                st.session_state["prediction_results"] = []
                st.session_state["prediction_model"] = "none"
            else:
                results, model_name = predict_next_words(
                    tuple(words[-3:]),
                    limit=5,
                )
                st.session_state["prediction_results"] = results
                st.session_state["prediction_model"] = model_name

        except Exception as exc:
            st.error(
                "Could not download or open the database. "
                "Check the Hugging Face repository, filename, "
                "and Streamlit deployment logs."
            )
            st.exception(exc)
            st.session_state["prediction_results"] = []
            st.session_state["prediction_model"] = "none"


if st.session_state["prediction_results"]:
    st.subheader("Top 5 next-word suggestions")
    st.caption(f"Model used: {st.session_state['prediction_model']}")

    for index, item in enumerate(st.session_state["prediction_results"]):
        word = item["word"]
        a, b, c = st.columns([3, 1.5, 1.5])

        with a:
            st.markdown(f"**{word}**")
            st.caption(f"Count: {item['count']:,}")

        with b:
            st.metric("Probability", f"{item['probability']:.2%}")

        with c:
            st.button(
                "Use word",
                key=f"use_{index}_{word}",
                on_click=append_suggestion,
                args=(word,),
                use_container_width=True,
            )

    st.info(
        "Choose a word, then click **Suggest next words** "
        "again to continue the sentence."
    )

elif st.session_state["prediction_model"] == "none":
    st.info("No suggestions were found, or the database could not be loaded.")


st.divider()
st.caption(
    "Predictions are based on observed n-gram counts in the GPAC corpus. "
    "If a longer context has no matches, the app falls back to a shorter context."
)
