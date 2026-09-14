# 02 · Modern NLP with Hugging Face

Module 01 built attention from scratch. In practice, nobody retrains a
transformer from zero for every task — the Hugging Face ecosystem provides
pretrained transformers and a standard fine-tuning workflow. This module
covers tokenizers, loading a pretrained model, and fine-tuning it for
classification.

## Tokenizers: subword units, not whole words

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
text = "Transformers tokenize unfamiliarly-spelled words into subwords."
encoded = tokenizer(text, return_tensors="pt")
print(tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))
# ['[CLS]', 'transformers', 'token', '##ize', 'unfamiliar', '##ly', '-', 'spelled',
#  'words', 'into', 'sub', '##words', '.', '[SEP]']
print(encoded["input_ids"].shape)   # torch.Size([1, 14])
```

Rare or made-up words ("unfamiliarly") get split into known subword pieces
(`un`, `##familiar`, `##ly`) rather than mapped to an out-of-vocabulary
token — this is why modern tokenizers rarely hit words they can't represent
at all.

## Loading a pretrained model and running inference

```python
from transformers import AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

inputs = tok(["This movie was fantastic!", "What a waste of time."], padding=True, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
probs = torch.softmax(logits, dim=-1)
labels = [model.config.id2label[p.argmax().item()] for p in probs]
print(labels)   # ['POSITIVE', 'NEGATIVE']
print(probs.round(decimals=3))
```

## Fine-tuning for a new classification task

```python
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

dataset = load_dataset("imdb")
small_train = dataset["train"].shuffle(seed=42).select(range(2000))
small_test = dataset["test"].shuffle(seed=42).select(range(500))

base_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_fn(batch):
    return base_tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

train_tok = small_train.map(tokenize_fn, batched=True)
test_tok = small_test.map(tokenize_fn, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}

args = TrainingArguments(
    output_dir="./results", num_train_epochs=2, per_device_train_batch_size=16,
    eval_strategy="epoch", learning_rate=2e-5, weight_decay=0.01,
)

trainer = Trainer(
    model=model, args=args, train_dataset=train_tok, eval_dataset=test_tok,
    compute_metrics=compute_metrics,
)
trainer.train()
metrics = trainer.evaluate()
print(metrics)   # {'eval_accuracy': ~0.89, 'eval_f1': ~0.89, ...}
```

## Worked example: inspecting what fine-tuning changed

```python
before = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
after = trainer.model

sample = base_tokenizer("A gripping, beautifully shot film.", return_tensors="pt")
with torch.no_grad():
    before_logits = before(**sample).logits
    after_logits = after(**sample).logits
print("before fine-tuning:", torch.softmax(before_logits, dim=-1).round(decimals=3))
print("after fine-tuning: ", torch.softmax(after_logits, dim=-1).round(decimals=3))
# before: close to [0.5, 0.5] -- the fresh classification head is untrained
# after:  strongly [~0.02, ~0.98] -- confidently positive
```

## Cheat sheet

| Task | Code |
|---|---|
| Load a tokenizer | `AutoTokenizer.from_pretrained(name)` |
| Load a model for classification | `AutoModelForSequenceClassification.from_pretrained(name, num_labels=k)` |
| Tokenize a batch | `tokenizer(texts, padding=True, truncation=True, return_tensors="pt")` |
| Fine-tune | `Trainer(model, args, train_dataset, eval_dataset).train()` |
| Reuse an existing head | Omit `num_labels` when the pretrained head already matches your task |

## How It Actually Works

**Subword tokenization (WordPiece/BPE) is built by iteratively merging the
most frequent adjacent symbol pairs in a large training corpus.** Starting
from individual characters, the algorithm repeatedly finds the pair of
adjacent symbols that co-occurs most often across the training corpus and
merges them into a new symbol, up to a fixed vocabulary size (e.g. 30,000
for `distilbert-base-uncased`). Common whole words end up as single tokens
(`transformers` survives intact) because they were frequent enough to earn
their own merged symbol, while rare words fragment into pieces that
individually appeared often enough (`token`, `##ize`) — the `##` marks "this
piece continues the previous token, no space before it." This is why
tokenization never truly fails on novel input: worst case, a word
decomposes all the way down to known characters, guaranteeing every string
has *some* representable encoding.

**`from_pretrained` restores both architecture and the weights learned
during large-scale pretraining, and `num_labels` mechanically determines
what gets replaced.** The downloaded checkpoint contains the transformer
backbone's weights (attention + feed-forward blocks, per Module 01) trained
via a self-supervised objective on massive text (e.g. masked-language
modeling: predict a randomly hidden word from context). Passing `num_labels
=2` tells `AutoModelForSequenceClassification` to attach a *fresh, randomly
initialized* linear classification head of the right output size on top of
the backbone's pooled output — exactly the "freeze backbone, replace head"
transfer-learning pattern from Level 2 Module 06, except here the entire
backbone (not just the head) typically continues training too (full
fine-tuning), because `Trainer`'s default optimizer updates every parameter
unless explicitly frozen.

**Fine-tuning is standard gradient descent (Module 09) with a much smaller
learning rate on a much larger pretrained model.** `learning_rate=2e-5` is
roughly 100x smaller than typical training-from-scratch rates, for the same
reason Level 2 Module 06 used small rates when unfreezing pretrained
layers: the backbone's weights already encode broadly useful language
representations from pretraining, and large updates would overwrite that
knowledge before the fresh classification head has learned to exploit it.
The "before" output being near `[0.5, 0.5]` is a direct, mechanical
consequence of the classification head's *random initialization* — an
untrained linear layer produces essentially arbitrary logits, which
`softmax` renders as a near-uniform distribution over 2 classes; after
`trainer.train()` runs gradient descent for 2 epochs over 2000 labeled
examples, the head's weights (and to a lesser extent the fine-tuned
backbone) have shifted specifically to separate positive from negative
sentiment, producing the confident `[0.02, 0.98]` output.

## Exercise

Repeat the fine-tuning run with the backbone frozen (`for p in
model.distilbert.parameters(): p.requires_grad = False` before
`trainer.train()`), training only the classification head. Compare
`eval_accuracy`/`eval_f1` against full fine-tuning, and connect any gap to
the "backbone already encodes useful representations vs. task-specific
adaptation" distinction discussed above.
