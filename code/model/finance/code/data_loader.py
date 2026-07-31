"""
Data loading utilities for subreddit metadata files
"""

import json
import gzip
from pathlib import Path
from typing import Dict, List, Iterator, Optional
from dataclasses import dataclass


@dataclass
class SubredditMeta:
    """Basic subreddit metadata from meta-only file"""
    display_name: str
    id: str
    num_posts: int
    earliest_post: Optional[int] = None
    num_comments: Optional[int] = None


@dataclass 
class SubredditFull:
    """Full subreddit information from detailed file"""
    display_name: str
    id: str
    description: str
    public_description: str
    subscribers: Optional[int] = None
    active_user_count: Optional[int] = None
    num_posts: Optional[int] = None
    num_comments: Optional[int] = None
    over18: bool = False
    created: Optional[int] = None
    lang: Optional[str] = None


class SubredditDataLoader:
    """Load and parse subreddit metadata files"""
    
    def __init__(self, meta_file: str, full_file: str):
        self.meta_file = Path(meta_file)
        self.full_file = Path(full_file)
        
    def load_meta_file(self) -> Iterator[SubredditMeta]:
        """Load subreddit metadata from meta-only file (JSONL format)"""
        opener = gzip.open if self.meta_file.suffix == '.gz' else open
        
        with opener(self.meta_file, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    yield SubredditMeta(
                        display_name=data.get('display_name', ''),
                        id=data.get('id', ''),
                        num_posts=data.get('_meta', {}).get('num_posts', 0),
                        earliest_post=data.get('_meta', {}).get('earliest_post'),
                        num_comments=data.get('_meta', {}).get('num_comments')
                    )
    
    def load_full_file(self) -> Iterator[SubredditFull]:
        """Load full subreddit information from detailed file (JSONL format)"""
        opener = gzip.open if self.full_file.suffix == '.gz' else open
        
        with opener(self.full_file, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    yield SubredditFull(
                        display_name=data.get('display_name', ''),
                        id=data.get('id', ''),
                        description=data.get('description', ''),
                        public_description=data.get('public_description', ''),
                        subscribers=data.get('subscribers'),
                        active_user_count=data.get('active_user_count'),
                        num_posts=data.get('_meta', {}).get('num_posts'),
                        num_comments=data.get('_meta', {}).get('num_comments'),
                        over18=data.get('over18', False),
                        created=data.get('created'),
                        lang=data.get('lang')
                    )
    
    def load_meta_to_dict(self) -> Dict[str, SubredditMeta]:
        """Load meta file into dictionary keyed by display_name"""
        return {sub.display_name: sub for sub in self.load_meta_file()}
    
    def load_full_to_dict(self) -> Dict[str, SubredditFull]:
        """Load full file into dictionary keyed by display_name"""
        return {sub.display_name: sub for sub in self.load_full_file()}
    
    def merge_data(self, use_full_file: bool = True) -> Dict[str, SubredditFull]:
        """
        Merge meta and full data, prioritizing full data
        
        Args:
            use_full_file: If False, only use meta file (for memory efficiency with large files)
        """
        if use_full_file:
            print("Loading full data file (this may take time for large files)...")
            full_data = self.load_full_to_dict()
            print(f"Loaded {len(full_data)} subreddits from full file")
            
            meta_data = self.load_meta_to_dict()
            print(f"Loaded {len(meta_data)} subreddits from meta file")
            
            # Add any subreddits only in meta file
            for name, meta in meta_data.items():
                if name not in full_data:
                    full_data[name] = SubredditFull(
                        display_name=meta.display_name,
                        id=meta.id,
                        description='',
                        public_description='',
                        num_posts=meta.num_posts,
                        num_comments=meta.num_comments
                    )
            
            print(f"Merged total: {len(full_data)} subreddits")
            return full_data
        else:
            print("Using meta-only file (memory-efficient mode)")
            meta_data = self.load_meta_to_dict()
            
            # Convert meta to full format
            full_data = {}
            for name, meta in meta_data.items():
                full_data[name] = SubredditFull(
                    display_name=meta.display_name,
                    id=meta.id,
                    description='',
                    public_description='',
                    num_posts=meta.num_posts,
                    num_comments=meta.num_comments
                )
            
            print(f"Loaded {len(full_data)} subreddits from meta file")
            return full_data
