"""The labelled pair set the semantic cache's false-hit rate is measured on.

**Synthetic, and small.** 60 hand-written pairs. This is stated first because
it is the most important caveat in the repo: the false-hit rate measured here
is a property of these 60 pairs and this embedder, not of English, not of any
production workload, and not of any real embedding model. It is a
*methodology* demonstration with a real number attached, not a benchmark.

The set is built around one design rule: **the hard negatives must be
lexically close and semantically different.** A pair set of random unrelated
prompts measures nothing, because any threshold above zero scores perfectly on
it. The negatives here differ by a single entity, negation, unit, direction or
tense -- exactly the edits that a lexical or even a learned embedder is most
likely to miss, and exactly the edits that change the correct answer
completely.

That is why the measured false-hit rate in this repo is *high* rather than
reassuring. An adversarial pair set is supposed to produce an uncomfortable
number; a comfortable number would mean the set was too easy to be worth
measuring.
"""

from __future__ import annotations

from .cache import LabelledPair

# Paraphrases: different wording, same information need. A cache SHOULD serve
# one answer for both. Missing these is merely a wasted call.
EQUIVALENT: list[tuple[str, str]] = [
    ("How do I reverse a list in Python?", "What's the way to reverse a list in Python?"),
    ("What is the capital of France?", "Which city is the capital of France?"),
    ("Explain what a database index does", "What does a database index do?"),
    ("How do I center a div with CSS?", "What's the CSS to center a div?"),
    ("Summarise the causes of the French Revolution", "What caused the French Revolution?"),
    ("How can I read a JSON file in Python?", "What's the way to read JSON files in Python?"),
    ("What does HTTP status 404 mean?", "Meaning of the 404 HTTP status code?"),
    ("Write a regex to match an email address", "Give me a regular expression for emails"),
    ("How do I install packages with pip?", "What is the command to install a package via pip?"),
    ("What is the difference between TCP and UDP?", "How do TCP and UDP differ?"),
    ("Explain recursion to a beginner", "Can you explain recursion simply?"),
    ("How do I sort a dictionary by value?", "What's the way to sort a dict by its values?"),
    ("What is a foreign key in SQL?", "In SQL, what does a foreign key mean?"),
    ("How do I create a virtual environment?", "What's the command to make a virtualenv?"),
    ("Describe how garbage collection works", "How does garbage collection work?"),
    ("What is the boiling point of water?", "At what temperature does water boil?"),
    ("How do I undo the last git commit?", "What's the git command to undo my last commit?"),
    ("Explain the difference between a list and a tuple", "How does a list differ from a tuple?"),
    ("What is Big O notation?", "Can you explain Big O notation?"),
    ("How do I check if a file exists in Python?", "What's the way to test for a file's existence "
     "in Python?"),
    ("What does the SQL GROUP BY clause do?", "Explain the GROUP BY clause in SQL"),
    ("How do I merge two dictionaries?", "What's the way to combine two dicts?"),
    ("What is a race condition?", "Can you explain what a race condition is?"),
    ("How do I format a date in Python?", "What's the way to format dates in Python?"),
    ("Explain what DNS does", "What is the purpose of DNS?"),
    ("What is the time complexity of binary search?", "How fast is binary search in Big O terms?"),
    ("How do I remove duplicates from a list?", "What's the way to deduplicate a list?"),
    ("What is a REST API?", "Can you explain what REST APIs are?"),
    ("How do I catch an exception in Python?", "What's the way to handle exceptions in Python?"),
    ("What does the git rebase command do?", "Explain what git rebase does"),
]

# Hard negatives: lexically close, semantically different. A cache MUST NOT
# serve one answer for both. These are the false-hit traps.
NON_EQUIVALENT: list[tuple[str, str]] = [
    # Single-entity swaps -- one word apart, completely different answer.
    ("What is the capital of France?", "What is the capital of Finland?"),
    ("How do I reverse a list in Python?", "How do I reverse a string in Python?"),
    ("What is the population of Austria?", "What is the population of Australia?"),
    ("Convert 100 USD to EUR", "Convert 100 EUR to USD"),
    ("What does HTTP status 404 mean?", "What does HTTP status 403 mean?"),
    ("How do I install packages with pip?", "How do I uninstall packages with pip?"),
    ("What is the boiling point of water?", "What is the freezing point of water?"),
    ("Sort the list in ascending order", "Sort the list in descending order"),
    # Negation -- the classic embedding failure.
    ("How do I enable logging in Django?", "How do I disable logging in Django?"),
    ("Is Python a compiled language?", "Is Python not a compiled language?"),
    ("Show me files modified today", "Show me files not modified today"),
    ("How do I make this function synchronous?", "How do I make this function asynchronous?"),
    # Units and magnitudes.
    ("What is 50 kilometres in miles?", "What is 50 miles in kilometres?"),
    ("Set the timeout to 30 seconds", "Set the timeout to 30 minutes"),
    ("How much is 5 GB in MB?", "How much is 5 MB in GB?"),
    # Direction and role reversal.
    ("How do I migrate from MySQL to Postgres?", "How do I migrate from Postgres to MySQL?"),
    ("Translate this from English to German", "Translate this from German to English"),
    ("How do I encrypt a file with GPG?", "How do I decrypt a file with GPG?"),
    ("Convert CSV to JSON", "Convert JSON to CSV"),
    # Version and tense -- same topic, different correct answer.
    ("What is new in Python 3.11?", "What is new in Python 3.12?"),
    ("How did the algorithm work in version 1?", "How does the algorithm work in version 2?"),
    ("What were the results last quarter?", "What are the projections next quarter?"),
    # Same domain, different question -- topically near, factually distinct.
    ("What is the time complexity of binary search?", "What is the space complexity of binary "
     "search?"),
    ("How do I undo the last git commit?", "How do I amend the last git commit?"),
    ("What is a foreign key in SQL?", "What is a primary key in SQL?"),
    ("Explain the difference between a list and a tuple", "Explain the difference between a list "
     "and a set"),
    ("How do I center a div horizontally?", "How do I center a div vertically?"),
    ("What is the difference between TCP and UDP?", "What is the difference between TCP and IP?"),
    ("How do I read a JSON file in Python?", "How do I write a JSON file in Python?"),
    ("What does git rebase do?", "What does git reset do?"),
]


def labelled_pairs() -> list[LabelledPair]:
    """The full set, positives first. Deterministic order, no sampling.

    Fixed rather than randomly sampled so the measured rate is reproducible
    byte-for-byte in CI -- a false-hit rate that moves between runs cannot be
    committed to a results file or defended in an ADR.
    """
    out = [LabelledPair(a, b, True) for a, b in EQUIVALENT]
    out += [LabelledPair(a, b, False) for a, b in NON_EQUIVALENT]
    return out


def counts() -> tuple[int, int]:
    """(equivalent, non-equivalent) sizes, for provenance in results."""
    return len(EQUIVALENT), len(NON_EQUIVALENT)
