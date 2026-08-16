"""
Main pipeline for financial subreddit extraction
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from data_loader import SubredditDataLoader, SubredditFull
from keyword_filter import KeywordFilter, KeywordMatch
from llm_annotator import LLMAnnotator, LLMAnnotation
from filter_rank import FilterRank, RankedSubreddit
from manual_review import ManualReview, ReviewItem
from user_sampler import UserSampler, SamplingConfig
from arctic_shift_client import ArcticShiftClient
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


class FinancialSubredditPipeline:
    """Complete pipeline for extracting and validating financial subreddits"""
    
    def __init__(self, meta_file: str, full_file: str, 
                 output_dir: str = "output",
                 min_posts: int = 1000,
                 min_subscribers: int = 100,
                 min_keywords: int = 1,
                 use_description: bool = True,
                 use_full_file: bool = True):
        
        self.meta_file = meta_file
        self.full_file = full_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.min_posts = min_posts
        self.min_subscribers = min_subscribers
        self.min_keywords = min_keywords
        self.use_description = use_description
        self.use_full_file = use_full_file
        
        # Initialize components
        self.loader = SubredditDataLoader(meta_file, full_file)
        self.keyword_filter = KeywordFilter()
        self.filter_rank = FilterRank(config.KNOWN_FINANCIAL_SUBREDDITS)
        
        # Results storage
        self.all_subreddits: List[SubredditFull] = []
        self.filtered_subreddits: List[SubredditFull] = []
        self.keyword_matches: List[KeywordMatch] = []
        self.ranked_subreddits: List[RankedSubreddit] = []
        self.final_subreddits: List[str] = []
    
    def stage1_basic_filtering(self) -> List[SubredditFull]:
        """Stage 1: Filter by activity thresholds"""
        logger.info("Stage 1: Basic filtering by activity thresholds")
        
        # Load all data
        logger.info("Loading subreddit data...")
        self.all_subreddits = list(self.loader.merge_data(use_full_file=self.use_full_file).values())
        logger.info(f"Loaded {len(self.all_subreddits)} subreddits")
        
        # Apply activity filters
        logger.info(f"Filtering by min_posts={self.min_posts}, min_subscribers={self.min_subscribers}")
        self.filtered_subreddits = self.filter_rank.filter_by_activity(
            self.all_subreddits,
            min_posts=self.min_posts,
            min_subscribers=self.min_subscribers
        )
        
        logger.info(f"After filtering: {len(self.filtered_subreddits)} subreddits")
        
        # Save intermediate results
        self._save_stage_results(self.filtered_subreddits, "stage1_filtered.json")
        
        return self.filtered_subreddits
    
    def stage2_keyword_matching(self, subreddits: List[SubredditFull]) -> List[SubredditFull]:
        """Stage 2: Keyword-based filtering"""
        logger.info("Stage 2: Keyword-based filtering")
        logger.info(f"Using description field: {self.use_description}")
        logger.info(f"Minimum keywords required: {self.min_keywords}")
        
        # Apply keyword filter
        logger.info("Matching financial keywords...")
        self.keyword_matches = self.keyword_filter.filter_batch(subreddits, self.use_description)
        
        # Get financial subreddits
        financial_subs = self.keyword_filter.get_financial_subreddits(
            subreddits, 
            min_keywords=self.min_keywords,
            use_description=self.use_description
        )
        
        logger.info(f"Found {len(financial_subs)} potentially financial subreddits")
        
        # Save intermediate results
        self._save_keyword_matches()
        self._save_stage_results(financial_subs, "stage2_keyword_matched.json")
        
        return financial_subs
    
    def stage3_ranking(self, subreddits: List[SubredditFull]) -> List[RankedSubreddit]:
        """Stage 3: Rank subreddits by keyword count and subscribers"""
        logger.info("Stage 3: Ranking subreddits by keyword count and subscribers")
        
        self.ranked_subreddits = self.filter_rank.rank_subreddits(
            subreddits,
            self.keyword_matches
        )
        
        logger.info(f"Ranked {len(self.ranked_subreddits)} subreddits")
        
        # Save ranked results
        self._save_ranked_results()
        
        return self.ranked_subreddits
    
    def stage5_manual_review(self, ranked: List[RankedSubreddit]) -> List[str]:
        """Stage 5: Manual review of edge cases"""
        logger.info("Stage 5: Manual review")
        
        # Prepare review items (focus on medium confidence)
        edge_cases = [r for r in ranked if 0.4 <= r.llm_confidence < 0.8]
        logger.info(f"Edge cases for review: {len(edge_cases)}")
        
        # Export for review
        review_items = self.manual_review.prepare_review_items(edge_cases, self.llm_annotations)
        self.manual_review.export_for_review(review_items, str(self.output_dir / "review_batch.json"))
        
        logger.info(f"Exported review batch to {self.output_dir / 'review_batch.json'}")
        logger.info("Please review the file and update decisions, then run import_review_decisions()")
        
        return []
    
    def stage6_user_sampling(self, subreddits: List[str], 
                            user_activity_data: dict,
                            strategy: str = "hybrid") -> dict:
        """Stage 6: Sample users from financial subreddits"""
        logger.info("Stage 6: User sampling")
        
        sampler = UserSampler(SamplingConfig())
        
        sampled_users = sampler.sample_from_multiple_subreddits(
            subreddits,
            user_activity_data,
            strategy=strategy
        )
        
        # Deduplicate
        unique_users = sampler.deduplicate_users(sampled_users)
        logger.info(f"Sampled {len(unique_users)} unique users")
        
        # Generate report
        report = sampler.generate_sampling_report(unique_users)
        logger.info(f"Sampling report: {report}")
        
        # Save results
        self._save_sampled_users(unique_users)
        
        return report
    
    def run_full_pipeline(self) -> List[str]:
        """Run complete pipeline"""
        logger.info("="*60)
        logger.info("FINANCIAL SUBREDDIT EXTRACTION PIPELINE")
        logger.info("="*60)
        
        # Stage 1: Basic filtering
        filtered = self.stage1_basic_filtering()
        
        # Stage 2: Keyword matching
        keyword_matched = self.stage2_keyword_matching(filtered)
        
        # Stage 3: Ranking
        ranked = self.stage3_ranking(keyword_matched)
        
        # Get final list (all ranked subreddits pass the keyword threshold)
        self.final_subreddits = [r.subreddit.display_name for r in ranked]
        
        # Save final results
        self._save_final_results()
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Final financial subreddits: {len(self.final_subreddits)}")
        logger.info(f"Results saved to {self.output_dir}")
        logger.info("="*60)
        
        return self.final_subreddits
    
    def _save_stage_results(self, subreddits: List[SubredditFull], filename: str):
        """Save intermediate results"""
        data = [s.__dict__ for s in subreddits]
        with open(self.output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_keyword_matches(self):
        """Save keyword match results"""
        data = [
            {
                "subreddit": km.subreddit,
                "matched_keywords": km.matched_keywords,
                "excluded_keywords": km.excluded_keywords,
                "keyword_count": km.keyword_count,
                "is_financial": km.is_financial
            }
            for km in self.keyword_matches
        ]
        with open(self.output_dir / "keyword_matches.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_llm_annotations(self):
        """Save LLM annotations"""
        data = [
            {
                "subreddit": la.subreddit,
                "is_financial": la.is_financial,
                "confidence": la.confidence,
                "reasoning": la.reasoning,
                "category": la.category,
                "error": la.error
            }
            for la in self.llm_annotations
        ]
        with open(self.output_dir / "llm_annotations.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_ranked_results(self):
        """Save ranked results"""
        data = [
            {
                "rank": r.rank,
                "subreddit": r.subreddit.display_name,
                "subscribers": r.subscribers,
                "num_posts": r.num_posts,
                "keyword_count": r.keyword_count,
                "is_known_financial": r.is_known_financial
            }
            for r in self.ranked_subreddits
        ]
        with open(self.output_dir / "ranked_subreddits.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_final_results(self):
        """Save final validated subreddits"""
        with open(self.output_dir / "financial_subreddits_final.json", 'w', encoding='utf-8') as f:
            json.dump(self.final_subreddits, f, indent=2)
    
    def _save_sampled_users(self, users):
        """Save sampled users"""
        data = [u.__dict__ for u in users]
        with open(self.output_dir / "sampled_users.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
