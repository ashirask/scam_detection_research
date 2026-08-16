"""
Example usage of the financial subreddit extraction pipeline
"""

import argparse
from pipeline import FinancialSubredditPipeline
from user_sampler import UserSampler, SamplingConfig
from arctic_shift_client import ArcticShiftClient


def example_basic_pipeline(min_keywords=1, use_description=True, use_full_file=False):
    """Run basic pipeline with simplified filtering"""
    # Parameters:
    # min_posts: Minimum posts required (default: 1000)
    # min_subscribers: Minimum subscribers required (default: 100)
    # min_keywords: Minimum financial keywords required (default: 1)
    # use_description: If True, search description field; if False, only public_description
    # use_full_file: If False, only load meta file (memory efficient)
    
    pipeline = FinancialSubredditPipeline(
        meta_file="subreddits_meta_only_2025-01",
        full_file="subreddits_2025-01",
        output_dir="output",
        min_posts=1000,
        min_subscribers=100,
        min_keywords=min_keywords,
        use_description=use_description,
        use_full_file=use_full_file
    )
    
    # Run pipeline
    financial_subs = pipeline.run_full_pipeline()
    
    print(f"Found {len(financial_subs)} financial subreddits")
    return financial_subs


def example_with_llm():
    """Run full pipeline with LLM annotation"""
    import os
    
    pipeline = FinancialSubredditPipeline(
        meta_file="subreddits_meta_only_2025-01",
        full_file="subreddits_2025-01",
        output_dir="output",
        min_posts=1000,
        openai_api_key=os.getenv("OPENAI_API_KEY")  # Requires API key
    )
    
    # Run full pipeline with LLM
    financial_subs = pipeline.run_full_pipeline(skip_llm=False)
    
    print(f"Found {len(financial_subs)} financial subreddits")
    return financial_subs


def example_user_sampling():
    """Example of user sampling from financial subreddits"""
    # Assume you have a list of financial subreddits
    financial_subreddits = ["personalfinance", "investing", "stocks", "cryptocurrency"]
    
    # Mock user activity data (in practice, get this from Arctic Shift or Academic Torrents)
    user_activity_data = {
        "personalfinance": {
            "user1": 150,
            "user2": 89,
            "user3": 45,
            # ... more users
        },
        "investing": {
            "user4": 200,
            "user5": 120,
            # ... more users
        }
    }
    
    sampler = UserSampler(SamplingConfig(users_per_subreddit=50))
    
    # Sample users using hybrid strategy
    sampled_users = sampler.sample_from_multiple_subreddits(
        financial_subreddits,
        user_activity_data,
        strategy="hybrid"
    )
    
    # Deduplicate users across subreddits
    unique_users = sampler.deduplicate_users(sampled_users)
    
    # Generate report
    report = sampler.generate_sampling_report(unique_users)
    print(f"Sampling report: {report}")
    
    return unique_users


def example_arctic_shift_extraction():
    """Example of extracting data using Arctic Shift API"""
    client = ArcticShiftClient()
    
    # Extract data for a specific subreddit
    result = client.extract_subreddit_data(
        subreddit="personalfinance",
        output_dir="output/personalfinance",
        include_posts=True,
        include_comments=True
    )
    
    print(f"Extracted {result['num_posts']} posts and {result['num_comments']} comments")
    
    # Extract data for specific users
    usernames = ["user1", "user2", "user3"]
    user_results = client.extract_user_data(usernames, output_dir="output/users")
    
    print(f"Extracted data for {len(user_results)} users")
    return user_results


def example_manual_review():
    """Example of manual review workflow"""
    from manual_review import ManualReview
    
    reviewer = ManualReview("output/review_results.json")
    
    # Import reviewed decisions
    reviewed_items = reviewer.import_review_decisions("output/review_batch.json")
    
    # Get approved subreddits
    approved = reviewer.get_approved(reviewed_items)
    rejected = reviewer.get_rejected(reviewed_items)
    
    print(f"Approved: {len(approved)}, Rejected: {len(rejected)}")
    
    # Generate summary
    summary = reviewer.generate_summary(reviewed_items)
    print(f"Summary: {summary}")
    
    return approved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial subreddit extraction pipeline")
    parser.add_argument("--meta-file", type=str, default="subreddits_meta_only_2025-01",
                       help="Path to meta-only subreddit file (default: subreddits_meta_only_2025-01)")
    parser.add_argument("--full-file", type=str, default="subreddits_2025-01",
                       help="Path to full subreddit file (default: subreddits_2025-01)")
    parser.add_argument("--output-dir", type=str, default="output",
                       help="Output directory (default: output)")
    parser.add_argument("--min-posts", type=int, default=1000,
                       help="Minimum posts required (default: 1000)")
    parser.add_argument("--min-subscribers", type=int, default=100,
                       help="Minimum subscribers required (default: 100)")
    parser.add_argument("--min-keywords", type=int, default=1, 
                       help="Minimum number of financial keywords required (default: 1)")
    parser.add_argument("--use-description", action="store_true", default=True,
                       help="Search description field (default: True)")
    parser.add_argument("--no-description", action="store_false", dest="use_description",
                       help="Do not search description field, only public_description")
    parser.add_argument("--use-full-file", action="store_true", default=False,
                       help="Load full data file instead of meta-only (default: False)")
    
    args = parser.parse_args()
    
    print(f"Running pipeline with:")
    print(f"  Meta file: {args.meta_file}")
    print(f"  Full file: {args.full_file}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Minimum posts: {args.min_posts}")
    print(f"  Minimum subscribers: {args.min_subscribers}")
    print(f"  Minimum keywords: {args.min_keywords}")
    print(f"  Use description field: {args.use_description}")
    print(f"  Use full file: {args.use_full_file}")
    
    pipeline = FinancialSubredditPipeline(
        meta_file=args.meta_file,
        full_file=args.full_file,
        output_dir=args.output_dir,
        min_posts=args.min_posts,
        min_subscribers=args.min_subscribers,
        min_keywords=args.min_keywords,
        use_description=args.use_description,
        use_full_file=args.use_full_file
    )
    
    financial_subs = pipeline.run_full_pipeline()
    print(f"Found {len(financial_subs)} financial subreddits")
