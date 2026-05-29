# Other Metrics Overview

This folder collects the faster, direct behavioral detectors we built for the scam-detection work.

## Active scripts

- [Fast replies](code/fast_replies/fast_reddit_parent_reply.py): direct parent-reply detector that flags accounts replying quickly to a comment within a time threshold. This is the cleanest signal for rapid reply behavior because it avoids sibling-reply ambiguity.
- [Co-URL detection](code/co-URL/README_detect_url_sharing_direct.md): direct TF-IDF detector that compares authors by the URLs they share. It supports post `domain` vs `full_url` modes, TF-IDF pruning, and observed-percentile thresholding.
- [Co-subreddit detection](code/co-subreddit/README_detect_subreddit_sharing.md): direct TF-IDF detector that compares users by the subreddits they post in. It mirrors the co-URL workflow but uses subreddit names as the token.
- [Co-submission detection](code/co-submission/README_detect_submission_sharing.md): direct TF-IDF detector that compares users by the submissions they comment on. It uses `link_id` as the token, which makes it a good fit for the extracted comments JSON produced by `extract_comments_for_posts.py`.

## Why these methods

These scripts focus on account-level behavior that is easier to interpret and validate than broad pairwise or null-model approaches.

- Fast replies capture direct comment-to-comment behavior for a single account.
- Co-URL compares authors by external resources they repeatedly post or mention.
- Co-subreddit compares users by the communities they participate in.
