"""Generate the tiny placeholder instruction set at data/sft_sample.jsonl.

This exists so the SFT script, the chat CLI and the web app can all be exercised
before any real instruction data is collected.  It is far too small to produce a
useful assistant - swap in a real dataset (Alpaca, Dolly, OpenAssistant, or your
own) via `--data` once the pipeline is proven.

    python scripts/make_sft_sample.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("data/sft_sample.jsonl")

PAIRS = [
    ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet, around 1595."),
    ("What is a sonnet?", "A sonnet is a fourteen-line poem, usually in iambic pentameter, with a fixed rhyme scheme."),
    ("Say hello.", "Hello! How can I help you today?"),
    ("Who are you?", "I am a small language model trained from scratch on a modest corpus."),
    ("What is iambic pentameter?", "It is a line of five iambs: five pairs of an unstressed syllable followed by a stressed one."),
    ("Name a tragedy by Shakespeare.", "Hamlet is one of his best known tragedies."),
    ("What is 2 + 2?", "2 + 2 is 4."),
    ("Translate 'thou art' into modern English.", "'Thou art' simply means 'you are'."),
    ("Give me a greeting in the style of a play.", "Good morrow, friend! What news do you bring?"),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("How many acts are in a Shakespeare play?", "Most of his plays are divided into five acts."),
    ("What does 'wherefore' mean?", "'Wherefore' means 'why', not 'where'."),
    ("Tell me something about Macbeth.", "Macbeth is a tragedy about a Scottish general whose ambition drives him to murder the king."),
    ("What is a soliloquy?", "A soliloquy is a speech in which a character speaks their thoughts aloud while alone on stage."),
    ("Write one line of poetry about the sea.", "The restless sea keeps counting out the shore."),
    ("What is a comedy?", "In Shakespeare's sense, a comedy is a play that ends in reconciliation or marriage rather than death."),
    ("Who is Hamlet's father?", "Hamlet's father is the late King of Denmark, who appears as a ghost."),
    ("What is the plural of 'thou'?", "The plural, and also the formal singular, is 'you'."),
    ("Say goodbye politely.", "Farewell, and may your day go well."),
    ("What is 10 divided by 2?", "10 divided by 2 is 5."),
    ("Explain what a language model does.", "A language model predicts the next token in a sequence, and generates text by doing that repeatedly."),
    ("What is training data?", "Training data is the text a model learns from by predicting each next token in it."),
    ("Who is Juliet?", "Juliet is the daughter of the Capulet family and one of the two lovers in Romeo and Juliet."),
    ("Give a short definition of a metaphor.", "A metaphor describes one thing as if it were another, without using 'like' or 'as'."),
    ("What colour is the sky on a clear day?", "On a clear day the sky is blue."),
    ("Count from one to five.", "One, two, three, four, five."),
    ("What is a stanza?", "A stanza is a grouped set of lines in a poem, separated from other groups by a blank line."),
    ("Say something encouraging.", "You are making good progress; keep going."),
    ("What is the opposite of 'ancient'?", "The opposite of 'ancient' is 'modern'."),
    ("Who directs a play?", "A director shapes the staging and performances of a play."),
]

SYSTEM = "You are a concise, helpful assistant."


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for i, (prompt, response) in enumerate(PAIRS):
            messages = []
            if i % 3 == 0:  # vary the shape so the model sees both forms
                messages.append({"role": "system", "content": SYSTEM})
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": response})
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
    print(f"wrote {len(PAIRS)} examples to {OUT}")


if __name__ == "__main__":
    main()
