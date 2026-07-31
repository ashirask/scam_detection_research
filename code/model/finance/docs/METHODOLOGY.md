# Detailed Methodology: Financial Subreddit Extraction Pipeline

## Overview
This pipeline extracts financial subreddits from Reddit metadata (January 2025) using a multi-stage filtering approach. The methodology combines keyword matching, activity-based filtering, and optional LLM semantic analysis to identify finance-related communities.

## File-by-File Methodology

### 1. config.py - Configuration Management
**Purpose**: Centralized configuration for all pipeline parameters

**Key Components**:
- File paths for input data (meta file, full file)
- Filtering thresholds (minimum posts, subscribers, active users)
- LLM configuration (model selection, batch size, temperature)
- Financial category definitions
- Known financial subreddit list (for auto-approval)

**Methodology**: All magic numbers and configuration parameters are centralized here to enable easy tuning without modifying core logic.

---

### 2. data_loader.py - Data Ingestion
**Purpose**: Load and parse Reddit metadata files

**Methodology**:
- **File Format Handling**: Supports both plain JSONL and gzipped JSONL formats
- **Data Structures**: 
  - `SubredditMeta`: Basic metadata (name, ID, post counts)
  - `SubredditFull`: Complete subreddit information (descriptions, subscribers, activity metrics)
- **Merging Strategy**: Combines meta-only and full data, prioritizing full data but preserving subreddits that only appear in meta file
- **Streaming**: Uses generators to handle large files without loading everything into memory

**Key Functions**:
- `load_meta_file()`: Streams through meta-only file line by line
- `load_full_file()`: Streams through detailed file line by line
- `merge_data()`: Combines both sources with conflict resolution

---

### 3. keyword_filter.py - Rule-Based Classification
**Purpose**: Initial financial relevance detection using keyword matching

**Methodology**:

**Keyword Categories**:
1. **Investing**: stock, stocks, invest, finance, market, trading, equity, portfolio, dividend
2. **Wall Street**: wallstreet, wsb, wallstreetbets
3. **Crypto**: crypto, bitcoin, ethereum, defi, blockchain, altcoin
4. **Trading Instruments**: options, futures, forex, etf, commodities
5. **General Finance**: money, budget, debt, credit, retirement, frugal, entrepreneur

**False Positive Removal**:
- Maintains exclusion list for non-financial uses of financial terms
- Examples: livestock, stockshow, marketplace, marketing, cryptozoology
- Any match with false positive keyword automatically gets score of 0

**Scoring Algorithm**:
```
base_score = min(matched_keywords * 0.2, 1.0)
if high_value_keyword_matched:
    base_score += 0.3
if false_positive_matched:
    score = 0.0
```

**Output**: `KeywordMatch` objects with matched keywords, excluded keywords, and relevance score

---

### 4. filter_rank.py - Multi-Criteria Ranking
**Purpose**: Combine multiple signals into unified ranking

**Methodology**:

**Activity Score Calculation**:
- Subscriber score: log-scale, max 0.3 points
- Post score: linear scale, max 0.3 points  
- Active user score: linear scale, max 0.4 points

**Combined Score Formula**:
```
combined_score = (keyword_score * 0.4) + (llm_confidence * 0.4) + (activity_score * 0.2)
if is_known_financial:
    combined_score += 0.2
```

**Ranking Strategy**:
- Sorts all subreddits by combined score (descending)
- Assigns rank positions
- Enables filtering by threshold (e.g., score >= 0.5)

**Key Functions**:
- `rank_subreddits()`: Combines keyword matches, LLM annotations, and activity data
- `filter_by_threshold()`: Returns subreddits above minimum score
- `filter_by_activity()`: Pre-filtering by activity metrics

---

### 5. llm_annotator.py - Semantic Analysis (Optional)
**Purpose**: Use LLM to understand context and meaning beyond keyword matching

**Methodology**:

**Prompt Engineering**:
- System prompt defines financial content classifier role
- Specifies categories: personal_finance, investing, crypto, trading, real_estate, etc.
- Instructs conservative approach (unsure = not financial)

**Annotation Process**:
1. Creates user prompt with subreddit name, description, public description
2. Sends to OpenAI API with JSON response format
3. Parses response to extract: is_financial, confidence, reasoning, category
4. Handles errors gracefully with fallback values

**Batch Processing**:
- Processes in configurable batch sizes (default: 50)
- Includes rate limiting (0.5 second delay between requests)
- Progress bar for long-running operations

**Output**: `LLMAnnotation` objects with confidence scores and categorization

---

### 6. manual_review.py - Human-in-the-Loop Validation
**Purpose**: Enable manual review of edge cases and final validation

**Methodology**:

**Review Workflow**:
1. **Export**: Converts ranked results to JSON for external review
2. **Review**: User manually approves/rejects each subreddit
3. **Import**: Loads reviewed decisions back into pipeline
4. **Validation**: Generates final approved list

**Interactive Mode**:
- Command-line interface for reviewing each item
- Shows: subreddit name, description, scores, LLM reasoning
- Commands: approve, reject, skip, quit
- Optional notes for each decision

**Statistics**:
- Total reviewed, approved, rejected, pending
- Approval rate calculation
- Breakdown by sampling reason

---

### 7. user_sampler.py - User Sampling Strategies
**Purpose**: Sample users from validated financial subreddits

**Methodology**:

**Sampling Strategies**:

1. **Top Activity**: Select users with highest post/comment counts
   - Prioritizes most active users
   - Good for identifying key community members

2. **Random**: Random selection from active users
   - Reduces bias toward hyper-active users
   - More representative sample

3. **Hybrid**: 70% top activity + 30% random
   - Balances activity and representation
   - Default recommended approach

**Activity Threshold**:
- Minimum activity requirement (default: 5 posts/comments)
- Filters out inactive or one-time users

**Deduplication**:
- Removes duplicate users across subreddits
- Keeps version with highest activity score

**Output**: `SampledUser` objects with sampling reason and activity scores

---

### 8. arctic_shift_client.py - Data Extraction
**Purpose**: Extract actual Reddit data via API or torrent

**Methodology**:

**Arctic Shift API**:
- Base URL: https://api.arctic-shift.com
- Endpoints: subreddit posts, comments, user activity
- Rate limiting: 100 requests/minute with automatic enforcement

**Data Types**:
- `PostData`: Post ID, author, title, content, score, timestamp
- `CommentData`: Comment ID, author, parent, body, score, timestamp

**Extraction Functions**:
- `get_subreddit_posts()`: Retrieve posts from specific subreddit
- `get_subreddit_comments()`: Retrieve comments from specific subreddit
- `get_user_activity()`: Get all activity for specific user
- `extract_subreddit_data()`: Complete extraction with file saving

**Academic Torrent Fallback**:
- Placeholder for torrent-based extraction
- Requires external torrent client integration

---

### 9. pipeline.py - Orchestration
**Purpose**: Coordinate all stages into cohesive workflow

**Methodology**:

**Pipeline Stages**:

**Stage 1: Basic Filtering**
- Loads all subreddit data
- Applies activity thresholds (posts, subscribers)
- Saves intermediate results
- Output: Filtered subreddit list

**Stage 2: Keyword Matching**
- Applies keyword filter to filtered subreddits
- Removes false positives
- Scores based on keyword relevance
- Output: Potentially financial subreddits

**Stage 3: LLM Annotation (Optional)**
- Runs LLM analysis on descriptions
- Generates confidence scores
- Categorizes by financial type
- Output: LLM annotations with confidence

**Stage 4: Ranking**
- Combines keyword scores, LLM confidence, activity
- Applies known subreddit bonus
- Sorts by combined score
- Output: Ranked subreddit list

**Stage 5: Manual Review**
- Exports edge cases for review
- Imports manual decisions
- Generates final approved list
- Output: Validated financial subreddits

**Stage 6: User Sampling**
- Samples users from validated subreddits
- Applies chosen sampling strategy
- Deduplicates across subreddits
- Output: Sampled user list

**Data Persistence**:
- Each stage saves intermediate results to JSON
- Enables debugging and stage-by-stage execution
- Final output: `financial_subreddits_final.json`

---

## Detailed Methodology for Choice 1 (Basic Pipeline - No LLM)

### example_basic_pipeline() Function

**Purpose**: Run pipeline using only keyword matching and activity filtering (no LLM costs)

**Step-by-Step Execution**:

1. **Initialization**
```python
pipeline = FinancialSubredditPipeline(
    meta_file="subreddits_meta_only_2025-01",
    full_file="subreddits_2025-01",
    output_dir="output",
    min_posts=1000,
    openai_api_key=None  # No LLM
)
```
- Sets up pipeline with data file paths
- Configures minimum post threshold (1000)
- Disables LLM by setting API key to None

2. **Pipeline Execution**
```python
financial_subs = pipeline.run_full_pipeline(skip_llm=True)
```

**Internal Pipeline Flow**:

**Stage 1: Basic Filtering**
- Reads both meta-only and full data files
- Merges data sources
- Filters out subreddits with < 1000 posts
- Filters out subreddits with < 100 subscribers
- Saves: `output/stage1_filtered.json`

**Stage 2: Keyword Matching**
- Extracts words from subreddit names and descriptions
- Matches against financial keyword list
- Removes matches that contain false positive keywords
- Calculates relevance score (0-1)
- Keeps subreddits with score >= 0.3
- Saves: `output/stage2_keyword_matched.json`
- Saves: `output/keyword_matches.json`

**Stage 3: LLM Annotation (SKIPPED)**
- Since `skip_llm=True`, this stage is bypassed
- No API calls made
- No costs incurred
- LLM confidence scores set to 0

**Stage 4: Ranking**
- Calculates activity scores (subscribers, posts, active users)
- Combines keyword score (40%) + LLM confidence (0%) + activity (20%)
- Since LLM is 0, effective weighting becomes: keyword (67%) + activity (33%)
- Sorts by combined score
- Saves: `output/ranked_subreddits.json`

**Stage 5: Manual Review (SKIPPED in basic mode)**
- Not automatically executed
- Can be run separately if needed

**Stage 6: Final Filtering**
- Filters to subreddits with combined score >= 0.5
- Extracts subreddit names
- Saves: `output/financial_subreddits_final.json`

3. **Output**
- Returns list of financial subreddit names
- Prints count and first 20 subreddits
- All intermediate results saved to `output/` directory

**Expected Results**:
- `financial_subreddits_final.json`: Final list of financial subreddits
- `stage1_filtered.json`: Subreddits passing activity filters
- `stage2_keyword_matched.json`: Subreddits passing keyword filter
- `keyword_matches.json`: Detailed keyword match information
- `ranked_subreddits.json`: All subreddits with scores and ranks

**Advantages of Choice 1**:
- No API costs
- Fast execution
- Transparent (keyword-based)
- Good starting point for understanding data

**Limitations**:
- May miss subreddits with non-obvious financial names
- Relies on keyword quality
- No semantic understanding of descriptions
- May have more false positives

**When to Use**:
- Initial exploration of data
- Budget constraints
- When keyword coverage is comprehensive
- For baseline comparison with LLM approach

---

## Usage Recommendations

**Start with Choice 1** to:
- Understand your data distribution
- Validate keyword coverage
- Establish baseline performance
- Identify gaps in keyword list

**Add LLM (Choice 2)** to:
- Improve semantic understanding
- Catch edge cases keywords miss
- Reduce false positives
- Get categorization by financial type

**Use Manual Review** to:
- Validate high-stakes decisions
- Handle edge cases
- Improve keyword/LLM prompts iteratively
- Build ground truth for future automation
