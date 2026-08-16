"""
Filtering and ranking logic for financial subreddits
"""

from typing import List
from dataclasses import dataclass
from data_loader import SubredditFull
from keyword_filter import KeywordMatch


@dataclass
class RankedSubreddit:
    """Subreddit with ranking information"""
    subreddit: SubredditFull
    keyword_count: int
    subscribers: int
    num_posts: int
    is_known_financial: bool
    rank: int = 0


class FilterRank:
    """Filter and rank subreddits based on simple criteria"""
    
    def __init__(self, known_financial: List[str]):
        self.known_financial = set(known_financial)
    
    def rank_subreddits(self, subreddits: List[SubredditFull], 
                       keyword_matches: List[KeywordMatch]) -> List[RankedSubreddit]:
        """Rank subreddits by keyword count and subscribers"""
        
        # Create lookup dictionary
        keyword_dict = {km.subreddit: km for km in keyword_matches}
        
        ranked = []
        for sub in subreddits:
            keyword_match = keyword_dict.get(sub.display_name)
            
            keyword_count = keyword_match.keyword_count if keyword_match else 0
            subscribers = sub.subscribers if sub.subscribers is not None else 0
            num_posts = sub.num_posts if sub.num_posts is not None else 0
            is_known = sub.display_name in self.known_financial
            
            ranked.append(RankedSubreddit(
                subreddit=sub,
                keyword_count=keyword_count,
                subscribers=subscribers,
                num_posts=num_posts,
                is_known_financial=is_known
            ))
        
        # Sort by keyword count (primary), then subscribers (secondary)
        ranked.sort(key=lambda x: (x.keyword_count, x.subscribers), reverse=True)
        
        # Assign ranks
        for i, r in enumerate(ranked):
            r.rank = i + 1
        
        return ranked
    
    def filter_by_activity(self, subreddits: List[SubredditFull],
                         min_posts: int = 1000,
                         min_subscribers: int = 100) -> List[SubredditFull]:
        """Filter subreddits by activity thresholds"""
        filtered = []
        for sub in subreddits:
            posts_ok = (sub.num_posts is not None and sub.num_posts >= min_posts)
            subs_ok = (sub.subscribers is not None and sub.subscribers >= min_subscribers)
            
            if posts_ok and subs_ok:
                filtered.append(sub)
        
        return filtered
