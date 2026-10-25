# Member 2: Trigram/Four-gram Models and Sentence Generation

Assignment tasks: **1.3** (top 10 n-grams, n = 3 and 4), **1.7** (sentence probability), **1.8** (random sentence generation).

## Files

| File | Contents |
|---|---|
| `trigram_fourgram_generation.ipynb` | All code (run in Google Colab, cells in order) |
| `top10_trigrams_full.csv` | Top 10 trigrams, full corpus |
| `top10_fourgrams_full.csv` | Top 10 four-grams, full corpus |
| `sentence_probabilities_full.csv` | log10 probability of test sentences under 5 models |
| `generated_sentences_full.csv` | 15 generated sentences with a copied/new check |

## Method

- **Preprocessing:** identical to Member 1 (same sentence-ending markers `፡፡ ። ፧ ? !` and punctuation regex; punctuation replaced by a space; whitespace tokenization; n-grams never cross sentence boundaries).
- **Consistency check:** full corpus gives 5,057,775 sentences and 77,760,019 tokens, exactly matching Member 1.
- **Probability:** MLE, same as Member 1's bigram table: `P(w | context) = count(context w) / outgoing_count(context)`.
- **Memory:** words are split into 8 fixed groups (CRC32 hash). Each pass over the corpus counts only n-grams whose first word is in one group. N-grams sharing a context land in the same group, so results are exact. About 19 minutes in Colab.

## Results (full corpus)

| | Trigrams | Four-grams |
|---|---|---|
| Unique | 42,137,242 | 45,110,256 |
| Total | 67,705,320 | 62,821,436 |
| Top n-gram | `እ ኤ አ` (16,708) | `ዓ ም ኢሳት ዜና` (9,680) |

Top n-grams are dominated by dates (`ቀን 2008 ዓ ም`), split abbreviations (`እ ኤ አ`, `ዓ ም`) and news boilerplate (`ኢሳት ዜና`, `የኔ ሃሳብ ዓምድ`).

### Sentence probability (log10)

| Sentence | MLE-2 | MLE-3 | MLE-4 | Laplace-4 | Interpolated |
|---|---|---|---|---|---|
| ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች። (given) | -inf | -inf | -inf | -17.478 | **-11.958** |
| ትገኛለች ቀንድ በአፍሪካ ኢትዮጵያ። (scrambled) | -inf | -inf | -inf | -23.291 | -17.486 |
| አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት። | -12.860 | -8.938 | -7.788 | -25.057 | -8.862 |
| ነገር ግን ይህ ብቻ አይደለም። | -8.645 | -6.947 | -7.098 | -17.441 | -7.585 |

- `ቀንድ ትገኛለች` never occurs in the full corpus (ቀንድ is usually followed by `አካባቢ`, `አገሮች`, `ላይ`, `ውስጥ`), so all plain MLE models give probability 0.
- Interpolation fixes the zero and rates the correct word order about 300,000 times more likely than the scrambled one.
- Laplace over-smooths because the vocabulary is large (V = 1,936,354).

### Sentence generation

Trained on every 10th sentence of the full corpus (505,778 sentences); copying checked against the full corpus.

| Model | Copied from corpus (of 5) |
|---|---|
| Bigram | 0 |
| Trigram | 0 |
| Four-gram | 5 (1 exact sentence) |

As n increases, sentences become more fluent but less original: the four-gram model mostly replays corpus text because most three-word contexts were seen only once.
