# Fast-reply detection: parent-reply methodology

Prefer the parent→child reply detector because it directly captures a single account's rapid reply behavior to an existing comment, which:

- Reduces false positives caused by multiple accounts independently replying to the same post (sibling replies).
- Maps naturally to an interpretable account-level signal (how often *this* account replies quickly), simplifying aggregation and ranking.
- Avoids combinatorial explosion of author pairs and the attendant normalization complexity.
- Makes it easier to inspect context (parent comment + immediate child) for human validation and triage.

Use the `fast_reddit_parent_reply.py` script to detect direct replies within a given time threshold (default 10s). Outputs include event-level CSVs and per-account summaries for follow-up analysis.
