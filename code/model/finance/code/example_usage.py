"""
Example usage of the financial subreddit extraction pipeline
"""

from pipeline import FinancialSubredditPipeline
from user_sampler import UserSampler, SamplingConfig
from arctic_shift_client import ArcticShiftClient


def example_basic_pipeline():
    """Run basic pipeline without LLM (keyword-only)"""
    # For large files (63GB+), use meta-only mode to save memory
    # Set use_full_file=False to only load the 500MB meta file
    # Set use_full_file=True to load both files (requires more memory)
    pipeline = FinancialSubredditPipeline(
        meta_file="subreddits_meta_only_2025-01",
        full_file="subreddits_2025-01",
        output_dir="output",
        min_posts=1000,
        openai_api_key=None,  # No LLM
        use_full_file=True   # Use meta-only for memory efficiency with large files
    )
    
    # Run pipeline with keyword filtering only
    financial_subs = pipeline.run_full_pipeline(skip_llm=True)
    
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
    print("Choose an example to run:")
    print("1. Basic pipeline (no LLM)")
    print("2. Full pipeline with LLM")
    print("3. User sampling")
    print("4. Arctic Shift data extraction")
    print("5. Manual review")
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == "1":
        example_basic_pipeline()
    elif choice == "2":
        example_with_llm()
    elif choice == "3":
        example_user_sampling()
    elif choice == "4":
        example_arctic_shift_extraction()
    elif choice == "5":
        example_manual_review()
    else:
        print("Invalid choice")
