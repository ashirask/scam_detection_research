# Financial Subreddit Extraction Pipeline

## Overview
This pipeline extracts financial subreddits from Reddit metadata (January 2025) using a multi-stage filtering approach combining keyword matching, LLM-based semantic analysis, and manual validation.

## Pipeline Stages

### Stage 1: Basic Filtering
- Filter subreddits by minimum post count (default: ≥1000 posts)
- Filter by activity metrics (subscribers, active users)
- Remove inactive/deleted subreddits

### Stage 2: Keyword Matching
- Match subreddit names against financial keywords
- Remove false positives using exclusion lists
- Score based on keyword relevance

### Stage 3: LLM Semantic Analysis
- Analyze subreddit descriptions for financial relevance
- Batch process for efficiency
- Generate confidence scores (0-1)
- Flag edge cases for manual review

### Stage 4: Known Subreddit Matching
- Match against curated list of known financial subreddits
- Auto-approve high-confidence matches

### Stage 5: Manual Review
- Interactive review interface for edge cases
- Batch approval/rejection capabilities
- Export final validated list

### Stage 6: User Sampling & Data Extraction
- Sample users from validated subreddits
- Integrate with Arctic Shift API for data extraction
- Alternative: Academic torrent download

## Data Sources

### Input Files
- `subreddits_meta_only_2025-01`: Basic metadata (display_name, id, post counts)
- `subreddits_2025-01`: Full subreddit information (descriptions, subscribers, etc.)

### Output Files
- `financial_subreddits_raw.json`: Initial filtered results
- `financial_subreddits_annotated.json`: LLM-annotated results
- `financial_subreddits_validated.json`: Manually validated final list
- `sampled_users.csv`: Users sampled from financial subreddits

## LLM Integration

### Configuration
- Model: OpenAI GPT-4 or similar (configurable)
- Batch size: 50 subreddits per batch
- Prompt engineering for financial relevance detection

### Annotation Format
```json
{
  "subreddit": "personalfinance",
  "is_financial": true,
  "confidence": 0.95,
  "reasoning": "Subreddit name and description clearly indicate personal finance focus",
  "category": "personal_finance"
}
```

## Usage

### Basic Pipeline
```python
from pipeline import FinancialSubredditPipeline

pipeline = FinancialSubredditPipeline(
    meta_file="subreddits_meta_only_2025-01",
    full_file="subreddits_2025-01",
    min_posts=1000
)

# Run full pipeline
results = pipeline.run_full_pipeline()

# Or run individual stages
filtered = pipeline.stage1_basic_filtering()
keyword_matched = pipeline.stage2_keyword_matching(filtered)
llm_annotated = pipeline.stage3_llm_annotation(keyword_matched)
```

### LLM Annotation Only
```python
from llm_annotator import LLMAnnotator

annotator = LLMAnnotator(api_key="your-key")
results = annotator.annotate_batch(subreddit_list)
```

## Dependencies
- Python 3.8+
- openai (for LLM)
- pandas
- requests (for Arctic Shift API)
- tqdm (progress bars)

## Arctic Shift API Integration
- Base URL: https://api.arctic-shift.com
- Endpoints: subreddit posts, comments, user activity
- Rate limiting: 100 requests/minute
