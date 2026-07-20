#Activity/volume features (easiest, do these first):
def activity_features(user_posts, user_comments):
    # num_submissions, num_comments, total_posts
    # account_age_days (days between first post and last post in your data)
    # submissions_per_day, comments_per_day
    # comment_to_submission_ratio
    # days_active (number of distinct days with any activity)

#Temporal features:
def temporal_features(user_comments, user_submissions):
    # mean_reply_time_seconds (time between parent post and this comment)
    # std_reply_time, min_reply_time (fast replies = bot signal)
    # hour_of_day entropy (bots often post at all hours uniformly)
    # day_of_week entropy
    # inter_post_interval_mean, inter_post_interval_std

#Text stylometric features (useful even without embeddings, inspired by the EACL paper):
def stylometric_features(texts):
    # mean_comment_length_chars, std_comment_length
    # mean_word_count
    # type_token_ratio (vocabulary diversity)
    # non_alpha_char_entropy  ← top feature in the EACL paper
    # uppercase_ratio
    # url_density (urls per post)
    # exclamation_density, question_density
    # unique_templates_ratio (how often exact same text appears)
    # repetition_ratio (what % of their posts are near-duplicates)


#Subreddit/URL diversity features (maps directly to your existing detectors):
def diversity_features(user_posts):
    # num_unique_subreddits
    # subreddit_entropy (are they posting everywhere or focused?)
    # num_unique_domains_shared
    # top_domain_concentration (are 90% of urls from one domain?)

#Embedding features (computationally heavy, do last):
def embedding_features(texts, model):
    # embed all texts with MPNet
    # mean_comment_embedding (384-dim vector) — flattened or PCA-reduced
    # mean_submission_embedding
    # intra_user_cosine_std (how diverse is their own content?)
    # This last one is very informative: low variance = likely templated bot


#Cross-user similarity features (most expensive, optional in first pass):
def similarity_features(user, sample_of_other_users):
    # for a sample of N=500-1000 random other users:
    #   compute text cosine sim to their mean embedding
    #   fraction_above_threshold_0.9
    # shared_url_overlap_score (from your existing co-URL TF-IDF)
    # For tractability: precompute a FAISS index on mean embeddings,
    #   then for each user do a range search. Much faster than pairwise.


#Username features (cheap signal, worth including):
def username_features(username):
    # username_length
    # has_digits (bots often end in numbers)
    # digit_ratio
    # has_bot_adjacent_word (b-words not exactly 'bot': 'bottle', etc.)
    # has_underscore, has_numbers_at_end
    # entropy of characters (random-looking names are often bots)
    
#Output: features.parquet with columns [author, feat_1, feat_2, ...] — one row per user, all numeric.