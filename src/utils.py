import os
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_github_token():
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        raise ValueError("GITHUB_TOKEN environment variable not set.")
    return token

def save_json(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def timestamp_str():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def handle_api_error(e):
    logging.error(f"API error: {str(e)}")
    if 'rate limit' in str(e).lower():
        logging.info("Rate limit hit. Consider waiting or using a different token.")
    raise e