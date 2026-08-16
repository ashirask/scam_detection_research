"""
Keyword-based filtering for financial subreddits with false positive removal
"""

from typing import Set, List, Dict, Tuple
from dataclasses import dataclass
from data_loader import SubredditFull


@dataclass
class KeywordMatch:
    """Result of keyword matching"""
    subreddit: str
    matched_keywords: List[str]
    excluded_keywords: List[str]
    keyword_count: int
    is_financial: bool


class KeywordFilter:
    """Filter subreddits based on financial keywords with false positive removal"""
    
    def __init__(self):
        # Financial keywords by category
        self.investing_keywords = {
            'stock', 'stocks', 'invest', 'investing', 'investor', 
            'finance', 'financial', 'market', 'markets', 'trading', 
            'trader', 'equity', 'equities', 'portfolio', 'dividend',
            'valueinvest', 'growthinvest'
        }
        
        self.wallstreet_keywords = {
            'wallstreet', 'wall_street', 'wallstreetbets', 'wsb'
        }
        
        self.crypto_keywords = {
            'crypto', 'bitcoin', 'btc', 'ethereum', 'eth', 'altcoin',
            'defi', 'nft', 'web3', 'blockchain', 'solana', 'doge', 'xrp'
        }
        
        self.trading_keywords = {
            'futures', 'forex', 'fx', 'commodity',
            'commodities', 'etf', 'mutualfund'
        }
        
        self.general_finance_keywords = {
            'money', 'cash', 'wealth', 'rich', 'income', 'salary',
            'budget', 'debt', 'credit', 'loan', 'mortgage', 'retirement',
            'saving', 'savings', 'spending', 'frugal', 'entrepreneur',
            'business', 'startup', 'sidehustle'
        }
        
        # Combine all financial keywords
        self.financial_keywords = (
            self.investing_keywords | self.wallstreet_keywords | 
            self.crypto_keywords | self.trading_keywords | 
            self.general_finance_keywords
        )
        
        # False positive keywords (non-financial uses of financial terms)
        self.false_positive_keywords = {
            'livestock', 'stockshow', 'stockdog', 'stockdogs', 'stockcar',
            'stockcars', 'stockholm', 'stockton', 'stocking', 'stockings',
            'stockphoto', 'stockphotos', 'stockphotography', 'marketplace',
            'marketing', 'marketer', 'marketing101', 'farmersmarket',
            'fleamarket', 'supermarket', 'tradingcards', 'tradingcard',
            'pokemontrades', 'pokemontrading', 'gameswap', 'hardwareswap',
            'giftcardexchange', 'cryptozoology', 'cryptid', 'cryptids',
            'wallpaper', 'wallpapers', 'wallart', 'goldenretriever',
            'goldenretrievers', 'goldfish', 'goldendoodle', 'goldendoodles',
            'goldcoast', 'silverado', 'silversmith', 'bulldog', 'bulldogs',
            'pitbull', 'pitbulls', 'chicagobulls'
        }
    
    def _extract_words(self, text: str) -> Set[str]:
        """Extract lowercase words from text"""
        if not text:
            return set()
        words = set()
        for word in text.lower().split():
            # Remove common punctuation
            word = word.strip('.,!?-_()[]{}":;')
            if word:
                words.add(word)
        return words
    
    def filter_subreddit(self, subreddit: SubredditFull, use_description: bool = True) -> KeywordMatch:
        """Filter a single subreddit based on keywords
        
        Args:
            subreddit: Subreddit to filter
            use_description: If True, search description field; if False, only public_description
        """
        # Build text to search based on description flag
        if use_description:
            text_to_check = f"{subreddit.display_name} {subreddit.description} {subreddit.public_description}"
        else:
            text_to_check = f"{subreddit.display_name} {subreddit.public_description}"
        
        words = self._extract_words(text_to_check)
        
        matched_keywords = []
        excluded_keywords = []
        
        # Check for financial keywords
        for word in words:
            if word in self.false_positive_keywords:
                excluded_keywords.append(word)
            elif word in self.financial_keywords:
                matched_keywords.append(word)
        
        is_financial = len(matched_keywords) > 0 and not excluded_keywords
        
        return KeywordMatch(
            subreddit=subreddit.display_name,
            matched_keywords=matched_keywords,
            excluded_keywords=excluded_keywords,
            keyword_count=len(matched_keywords),
            is_financial=is_financial
        )
    
    def filter_batch(self, subreddits: List[SubredditFull], use_description: bool = True) -> List[KeywordMatch]:
        """Filter a batch of subreddits
        
        Args:
            subreddits: List of subreddits to filter
            use_description: If True, search description field; if False, only public_description
        """
        results = []
        for subreddit in subreddits:
            result = self.filter_subreddit(subreddit, use_description)
            results.append(result)
        return results
    
    def get_financial_subreddits(self, subreddits: List[SubredditFull], min_keywords: int = 1, 
                                 use_description: bool = True) -> List[SubredditFull]:
        """Get only financial subreddits above keyword threshold
        
        Args:
            subreddits: List of subreddits to filter
            min_keywords: Minimum number of financial keywords required (default: 1)
            use_description: If True, search description field; if False, only public_description
        """
        matches = self.filter_batch(subreddits, use_description)
        subreddit_dict = {s.display_name: s for s in subreddits}
        
        financial = []
        for match in matches:
            if match.is_financial and match.keyword_count >= min_keywords:
                financial.append(subreddit_dict[match.subreddit])
        
        return financial
