# Sample evaluation set

46 short documents, 8 queries, graded relevance (2 = directly answers, 1 = relevant),
in BEIR layout. Every query has 4-5 relevant documents and at least one *lexical
distractor*: a document that shares the query's key word but is about something else
(iron skillets vs. iron deficiency, jet engines vs. jet lag, pythons the snake vs.
Python the language). Those are exactly the cases where a cross-encoder, which reads
the query and document together, should beat a single pooled vector.

It is small enough to inspect by eye and runs in seconds, which makes it useful for
tests and for intuition. It is far too small for a trustworthy precision number;
use BEIR NFCorpus (`make download-nfcorpus`) for the real evaluation.
