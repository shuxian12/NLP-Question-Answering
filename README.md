# NLP Question Answering

- **The SearchQA is a retrieval-based question-answer task challenge**

> `2022.11.25`
> This is my fist time writing the Question Answering, so there might be some mistakes, sorry!!

## Testing

<details>
<summary>Evulation Function</summary>

``` python
import re
import collections
import string

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
 
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)
 
    def white_space_fix(text):
        return " ".join(text.split())
 
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
 
    def lower(text):
        return text.lower()
 
    return white_space_fix(remove_articles(remove_punc(lower(s))))
 
def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
```
</details>

#### My Best Dev Result

|   F1   | Exact Match | All Match |
| :----: | :---------: | :-------: |
| 0.6511 |   0.60499   |  0.68764  |

#### Testing Score

|  F1   | Exact Match |
| :---: | :---------: |
| 0.45  |     0.4     |

## Model Architecture

  1. BERT
  2. Linear Layer ( sequence length to 2, for start and end, QA logits )
  3. Output processing
    `輸出時選擇top k的詞語組成輸出 (由於最後預測時一定會有解答，使用此方式避免預測無答案之輸出)`

## HyperParemeter
  
  ``` bash
  LEARNING_RATE = 1e-5  #(2e-5~1e-6)
  BATCH_SIZE = 16
  EPOCH = 5
  k = 3
  ```
