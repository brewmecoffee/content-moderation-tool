# Content Moderation Tool

A Python-based content moderation tool that uses DeepSeek's AI to analyze and filter posts for inappropriate content, specifically focusing on extreme profanity and explicit references.

## Features

- Automated content moderation using DeepSeek's AI API
- Batch processing of posts with progress tracking
- Graceful interruption handling with progress saving
- Customizable moderation criteria via prompt file
- Detailed logging of moderation decisions
- Optional web context lookup via Brave Search API

## Prerequisites

- Python 3.7+
- DeepSeek API key
- (Optional) Brave Search API key

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install aiohttp python-dotenv openai
```
3. Copy `.env.sample` to `.env` and add your API keys:
```
DEEPSEEK_API_KEY=your_deepseek_key
BRAVE_SEARCH_API_KEY=your_brave_search_key
```

## Usage

1. Create an input file (`posts.txt`) with your posts, separated by `---`:
```
First post content
---
Second post content
---
Third post content
```

2. (Optional) Customize moderation criteria in `moderation_prompt.txt`

3. Run the script:
```bash
python extract.py
```

The script will:
- Process all posts from `posts.txt`
- Save acceptable posts to `acceptable_posts.txt`
- Save flagged posts to `flagged_posts.txt`
- Maintain a detailed log in `moderation_log.json`
- Track progress in `moderation_progress.json`

## Configuration

Default settings can be modified in `.env`:
```
INPUT_FILE=posts.txt
KEEP_FILE=relevant_posts.txt
DISCARD_FILE=irrelevant_posts.txt
LOG_FILE=processing_log.json
```

## Interruption Handling

- The script can be safely interrupted with Ctrl+C
- Progress is automatically saved
- Processing can be resumed by running the script again

## Output Files

- `acceptable_posts.txt`: Posts that passed moderation
- `flagged_posts.txt`: Posts that were flagged as inappropriate
- `moderation_log.json`: Detailed log of all moderation decisions
- `moderation_progress.json`: Progress tracking for interrupted sessions

## Moderation Criteria

Posts are flagged if they contain:
- Explicit nudity references
- Extreme profanity
- Graphic sexual content
- Offensive slurs
- Violence-related content

See `moderation_prompt.txt` for detailed criteria.

## Error Handling

- Automatic retry with exponential backoff for API calls
- Progress saving on errors
- Detailed error logging

## License

[Add your license information here]

## Contributing

[Add contribution guidelines if applicable]