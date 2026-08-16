"""
Configuration for financial subreddit extraction pipeline
"""

# File paths (now configurable via command line, these are defaults)
META_FILE = "subreddits_meta_only_2025-01"
FULL_FILE = "subreddits_2025-01"
OUTPUT_DIR = "output"

# Filtering thresholds (now configurable via command line, these are defaults)
MIN_POSTS = 1000  # Minimum posts to consider a subreddit
MIN_SUBSCRIBERS = 100  # Minimum subscribers
MIN_ACTIVE_USERS = 10  # Minimum active users (no longer used in simplified pipeline)

# LLM Configuration
LLM_MODEL = "gpt-4"  # or "gpt-3.5-turbo" for faster/cheaper
LLM_BATCH_SIZE = 50
LLM_TEMPERATURE = 0.3  # Lower for more consistent results
LLM_MAX_TOKENS = 200

# Arctic Shift API
ARCTIC_SHIFT_BASE_URL = "https://api.arctic-shift.com"
ARCTIC_SHIFT_RATE_LIMIT = 100  # requests per minute

# Financial categories
FINANCIAL_CATEGORIES = [
    "personal_finance",
    "investing", 
    "stock_market",
    "crypto",
    "trading",
    "real_estate",
    "budgeting",
    "debt",
    "career_finance",
    "business",
    "other"
]

# Known financial subreddits (auto-approve)
KNOWN_FINANCIAL_SUBREDDITS = [
    "personalfinance",
    "financialindependence", 
    "leanfire",
    "fatFIRE",
    "investing",
    "stocks",
    "stockmarket",
    "wallstreetbets",
    "dividends",
    "options",
    "cryptocurrency",
    "bitcoin",
    "ethfinance",
    "realestate",
    "realestateinvesting",
    "frugal",
    "debtfree",
    "creditcards",
    "careerguidance"
]
