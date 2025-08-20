import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Any
from github import Github, GithubException, Repository, Commit, PullRequest
from github.PaginatedList import PaginatedList
from difflib import unified_diff
from src.utils import get_github_token, save_json, load_json, handle_api_error, timestamp_str

logger = logging.getLogger(__name__)

class HistoryParser:
    """
    Extracts GitHub repository history including commits, diffs, metadata, PRs, and comments.
    Supports caching to avoid repeated API calls. Handles rate limits and errors robustly.
    """
    def __init__(self, repo_url: str, cache_dir: str = 'data/cache', use_cache: bool = True):
        """
        Initialize parser with repo URL.
        :param repo_url: Full GitHub repo URL (e.g., https://github.com/user/repo)
        :param cache_dir: Directory for caching parsed data.
        :param use_cache: If True, load from cache if available.
        """
        self.repo_url = repo_url
        self.owner, self.repo_name = self._parse_repo_url(repo_url)
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.db_path = os.path.join(cache_dir, f"{self.owner}_{self.repo_name}.db")
        self.json_cache_path = os.path.join(cache_dir, f"{self.owner}_{self.repo_name}_history.json")
        os.makedirs(cache_dir, exist_ok=True)
        self.github = Github(get_github_token())
        self.repo: Repository = self._get_repo()

    def _parse_repo_url(self, url: str) -> tuple[str, str]:
        """Parse owner and repo name from URL."""
        if not url.startswith('https://github.com/'):
            raise ValueError("Invalid GitHub URL format.")
        parts = url.split('/')[3:5]
        if len(parts) != 2:
            raise ValueError("URL must be in format https://github.com/owner/repo")
        return parts[0], parts[1]

    def _get_repo(self) -> Repository:
        """Fetch repository object with error handling."""
        try:
            return self.github.get_repo(f"{self.owner}/{self.repo_name}")
        except GithubException as e:
            handle_api_error(e)

    def parse_history(self, max_commits: int = 1000, include_prs: bool = True) -> Dict[str, Any]:
        """
        Parse repo history into structured data.
        :param max_commits: Limit number of commits to fetch (for large repos).
        :param include_prs: If True, fetch PRs and comments.
        :return: Dict with commits, prs, metadata.
        """
        if self.use_cache and os.path.exists(self.json_cache_path):
            logger.info(f"Loading cached history from {self.json_cache_path}")
            return load_json(self.json_cache_path)

        data = {
            'metadata': self._get_repo_metadata(),
            'commits': self._get_commits(max_commits),
        }
        if include_prs:
            data['prs'] = self._get_prs()

        self._save_to_cache(data)
        return data

    def _get_repo_metadata(self) -> Dict[str, Any]:
        """Fetch basic repo metadata."""
        return {
            'name': self.repo.full_name,
            'description': self.repo.description,
            'stars': self.repo.stargazers_count,
            'forks': self.repo.forks_count,
            'created_at': self.repo.created_at.isoformat(),
            'updated_at': self.repo.updated_at.isoformat(),
        }

    def _get_commits(self, max_commits: int) -> List[Dict[str, Any]]:
        """Fetch commits with diffs and metadata, handling pagination."""
        commits = []
        try:
            paginated_commits: PaginatedList = self.repo.get_commits()
            for i, commit in enumerate(paginated_commits):
                if i >= max_commits:
                    break
                commit_data = self._parse_commit(commit)
                commits.append(commit_data)
                logger.info(f"Parsed commit {i+1}/{min(max_commits, paginated_commits.totalCount)}: {commit.sha}")
        except GithubException as e:
            handle_api_error(e)
        return commits

    def _parse_commit(self, commit: Commit) -> Dict[str, Any]:
        """Parse single commit into dict with diff."""
        files = []
        for file in commit.files:
            if file.patch:
                # Generate unified diff if needed
                diff = file.patch
            else:
                diff = ''  # Placeholder if no patch
            files.append({
                'filename': file.filename,
                'status': file.status,
                'additions': file.additions,
                'deletions': file.deletions,
                'changes': file.changes,
                'diff': diff,
            })
        return {
            'sha': commit.sha,
            'author': commit.author.login if commit.author else 'unknown',
            'date': commit.commit.author.date.isoformat(),
            'message': commit.commit.message,
            'files': files,
        }

    def _get_prs(self) -> List[Dict[str, Any]]:
        """Fetch PRs with comments and sentiment placeholder."""
        prs = []
        try:
            paginated_prs: PaginatedList = self.repo.get_pulls(state='all')
            for pr in paginated_prs:
                pr_data = {
                    'number': pr.number,
                    'title': pr.title,
                    'author': pr.user.login,
                    'created_at': pr.created_at.isoformat(),
                    'merged_at': pr.merged_at.isoformat() if pr.merged_at else None,
                    'comments': self._get_pr_comments(pr),
                }
                prs.append(pr_data)
        except GithubException as e:
            handle_api_error(e)
        return prs

    def _get_pr_comments(self, pr: PullRequest) -> List[Dict[str, Any]]:
        """Fetch comments for a PR."""
        comments = []
        for comment in pr.get_comments():
            comments.append({
                'author': comment.user.login,
                'body': comment.body,
                'created_at': comment.created_at.isoformat(),
                # Placeholder for sentiment (to be analyzed in miner)
                'sentiment': None,
            })
        return comments

    def _save_to_cache(self, data: Dict[str, Any]):
        """Save parsed data to JSON and SQLite for querying."""
        save_json(data, self.json_cache_path)
        logger.info(f"Saved history to {self.json_cache_path}")

        # Save to SQLite for efficient querying in later modules
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS commits
                          (sha TEXT PRIMARY KEY, author TEXT, date TEXT, message TEXT)''')
        for commit in data.get('commits', []):
            cursor.execute('INSERT OR REPLACE INTO commits VALUES (?, ?, ?, ?)',
                           (commit['sha'], commit['author'], commit['date'], commit['message']))
        conn.commit()
        conn.close()
        logger.info(f"Saved to SQLite: {self.db_path}")

    def get_cached_data(self) -> Dict[str, Any]:
        """Load data from cache if available."""
        return load_json(self.json_cache_path) or {}

# Example usage (for testing)
if __name__ == "__main__":
    parser = HistoryParser("https://github.com/octocat/Hello-World")
    history = parser.parse_history(max_commits=50)
    print(json.dumps(history, indent=2))