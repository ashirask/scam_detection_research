"""
Filtering and ranking logic for financial subreddits
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
from data_loader import SubredditFull
from keyword_filter import KeywordMatch
from llm_annotator import LLMAnnotation


@dataclass
class RankedSubreddit:
    """Subreddit with ranking information"""
    subreddit: SubredditFull
    keyword_score: float
    llm_confidence: float
    combined_score: float
    is_known_financial: bool
    rank: int = 0


class FilterRank:
    """Filter and rank subreddits based on multiple criteria"""
    
    def __init__(self, known_financial: List[str]):
        self.known_financial = set(known_financial)
    
    def _calculate_activity_score(self, subreddit: SubredditFull) -> float:
        """Calculate activity score based on subscribers and posts, normalizing for missing data"""
        score = 0.0
        
        # Track which fields are available
        available_fields = []
        
        # Subscriber score (30% weight if available)
        if subreddit.subscribers:
            available_fields.append('subscribers')
            score += min(subreddit.subscribers / 100000, 1.0) * 0.3
        
        # Post score (30% weight if available)
        if subreddit.num_posts:
            available_fields.append('posts')
            score += min(subreddit.num_posts / 50000, 1.0) * 0.3
        
        # Active user score (40% weight if available)
        if subreddit.active_user_count:
            available_fields.append('active_users')
            score += min(subreddit.active_user_count / 10000, 1.0) * 0.4
        
        # Normalize score if some fields are missing
        # This prevents penalizing subreddits with incomplete data
        if available_fields and len(available_fields) < 3:
            # Redistribute weight proportionally to available fields
            total_weight = sum([
                0.3 if 'subscribers' in available_fields else 0,
                0.3 if 'posts' in available_fields else 0,
                0.4 if 'active_users' in available_fields else 0
            ])
            if total_weight > 0:
                score = score / total_weight
        
        return score
    
    def _calculate_combined_score(self, keyword_score: float, llm_confidence: float, 
                                  activity_score: float, is_known: bool) -> float:
        """Calculate combined relevance score"""
        # Weight: 40% keyword, 40% LLM, 20% activity
        combined = (keyword_score * 0.4) + (llm_confidence * 0.4) + (activity_score * 0.2)
        
        # Bonus for known financial subreddits
        if is_known:
            combined = min(combined + 0.2, 1.0)
        
        return combined
    
    def rank_subreddits(self, subreddits: List[SubredditFull], 
                       keyword_matches: List[KeywordMatch],
                       llm_annotations: List[LLMAnnotation]) -> List[RankedSubreddit]:
        """Rank subreddits by combined relevance score"""
        
        # Create lookup dictionaries
        keyword_dict = {km.subreddit: km for km in keyword_matches}
        llm_dict = {la.subreddit: la for la in llm_annotations}
        
        ranked = []
        for sub in subreddits:
            keyword_match = keyword_dict.get(sub.display_name)
            llm_annotation = llm_dict.get(sub.display_name)
            
            keyword_score = keyword_match.score if keyword_match else 0.0
            llm_confidence = llm_annotation.confidence if llm_annotation else 0.0
            activity_score = self._calculate_activity_score(sub)
            is_known = sub.display_name in self.known_financial
            
            combined_score = self._calculate_combined_score(
                keyword_score, llm_confidence, activity_score, is_known
            )
            
            ranked.append(RankedSubreddit(
                subreddit=sub,
                keyword_score=keyword_score,
                llm_confidence=llm_confidence,
                combined_score=combined_score,
                is_known_financial=is_known
            ))
        
        # Sort by combined score
        ranked.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Assign ranks
        for i, r in enumerate(ranked):
            r.rank = i + 1
        
        return ranked
    
    def filter_by_threshold(self, ranked: List[RankedSubreddit], 
                           min_score: float = 0.5) -> List[RankedSubreddit]:
        """Filter ranked subreddits by minimum combined score"""
        return [r for r in ranked if r.combined_score >= min_score]
    
    def filter_by_activity(self, subreddits: List[SubredditFull],
                         min_posts: int = 1000,
                         min_subscribers: int = 100) -> List[SubredditFull]:
        """Filter subreddits by activity thresholds"""
        filtered = []
        for sub in subreddits:
            posts_ok = (sub.num_posts is None or sub.num_posts >= min_posts)
            subs_ok = (sub.subscribers is None or sub.subscribers >= min_subscribers)
            
            if posts_ok and subs_ok:
                filtered.append(sub)
        
        return filtered
    
    def get_top_n(self, ranked: List[RankedSubreddit], n: int) -> List[RankedSubreddit]:
        """Get top N ranked subreddits"""
        return ranked[:n]
