import os
import json
import time
import asyncio
import aiohttp
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
import openai  # Using OpenAI's client for DeepSeek's API (compatible format)
import random
import traceback
import signal
import sys

# Load environment variables
load_dotenv()

# API keys and configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")
BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

# File paths
INPUT_FILE = "posts.txt"
ACCEPTABLE_FILE = "acceptable_posts.txt"
FLAGGED_FILE = "flagged_posts.txt"
LOG_FILE = "moderation_log.json"
PROGRESS_FILE = "moderation_progress.json"
SYSTEM_PROMPT_FILE = "moderation_prompt.txt"  # File containing the system prompt

# API configuration
MAX_RETRIES = 5  # Maximum number of retry attempts for API calls
BASE_RETRY_DELAY = 2  # Base delay in seconds before retrying
MAX_RETRY_DELAY = 60  # Maximum retry delay in seconds
JITTER_FACTOR = 0.2  # Random jitter factor to add to retry delays

# Initialize DeepSeek client using OpenAI's client (compatible)
client = openai.OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1",  # DeepSeek's API endpoint
    timeout=60  # Increase timeout to handle potential slow responses
)

# Models available with DeepSeek
DEEPSEEK_MODELS = {
    "deepseek-chat": "DeepSeek V3 (general purpose)",
    "deepseek-reasoner": "DeepSeek R1 (advanced reasoning)"
}

# Select model to use
MODEL_TO_USE = "deepseek-chat"  # Using DeepSeek-V3 by default

# Flag to track if the program is being gracefully shut down
is_shutting_down = False

# Progress tracking
current_progress = {
    "completed_posts": [],
    "pending_posts": [],
    "total_posts": 0,
    "last_processed_index": -1
}

def load_system_prompt():
    """Load the system prompt from file."""
    if not os.path.exists(SYSTEM_PROMPT_FILE):
        print(f"System prompt file {SYSTEM_PROMPT_FILE} not found. Creating default one.")
        create_default_system_prompt()
    
    try:
        with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
            prompt = f.read()
        return prompt
    except Exception as e:
        print(f"Error loading system prompt: {e}")
        # Return a minimal fallback prompt if loading fails
        return "You are a content moderator. Determine if posts contain extreme profanity or nudity references."

def create_default_system_prompt():
    """Create the default system prompt file if it doesn't exist."""
    default_prompt = """You are a content moderation expert specializing in identifying inappropriate social media posts. Your task is to evaluate each post and determine if it should be "flagged" (contains extreme profanity or nudity references) or is "acceptable" (doesn't contain extreme profanity or nudity references).

FLAGGED CONTENT: Flag posts as "inappropriate" if they contain ANY of these elements:

1. EXPLICIT NUDITY REFERENCES
   - Explicit descriptions of nudity or naked body parts
   - Detailed sexual content or explicit sexual invitations
   - Sexualized descriptions of people, especially those that objectify them
   - Content that appears to solicit or offer sexual services

2. EXTREME PROFANITY
   - The most severe profanity (consider cultural context)
   - Graphic sexual terms used in a vulgar context
   - Extremely offensive slurs related to race, gender, sexuality, disability, etc.
   - Detailed descriptions of extreme violence or sexual violence
   - Content that advocates for harm to individuals or groups

ACCEPTABLE CONTENT: Content that may contain:
   - Mild profanity (common swear words used casually)
   - Non-explicit adult humor
   - General references to dating or relationships
   - Mild insults that don't contain slurs or hate speech
   - Discussion of mature themes without explicit details

You must return a JSON object with two fields:
- "acceptable": boolean (true if the post is acceptable, false if it should be flagged)
- "explanation": string explaining your reasoning in detail, including which category makes it inappropriate

BE THOROUGH in your evaluation. If you're unsure about a piece of content, err on the side of caution and flag it."""

    try:
        with open(SYSTEM_PROMPT_FILE, "w", encoding="utf-8") as f:
            f.write(default_prompt)
        print(f"Default system prompt created at {SYSTEM_PROMPT_FILE}")
    except Exception as e:
        print(f"Error creating default system prompt: {e}")

def save_progress():
    """Save current progress to file."""
    try:
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(current_progress, f, indent=2)
        print(f"Progress saved to {PROGRESS_FILE}")
    except Exception as e:
        print(f"Error saving progress: {e}")

def load_progress() -> bool:
    """Load progress from file if it exists.
    Returns True if progress was loaded, False otherwise."""
    global current_progress
    
    if not os.path.exists(PROGRESS_FILE):
        print("No previous progress found.")
        return False
    
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            saved_progress = json.load(f)
            
        if (isinstance(saved_progress, dict) and 
            "completed_posts" in saved_progress and 
            "pending_posts" in saved_progress and
            "total_posts" in saved_progress and
            "last_processed_index" in saved_progress):
            
            current_progress = saved_progress
            print(f"Loaded progress: {len(current_progress['completed_posts'])} posts processed, {len(current_progress['pending_posts'])} pending")
            return True
        else:
            print("Invalid progress file format. Starting fresh.")
            return False
    except Exception as e:
        print(f"Error loading progress: {e}")
        return False

def signal_handler(sig, frame):
    """Handle interrupt signals to save progress before exiting."""
    global is_shutting_down
    print("\nReceived interrupt signal. Saving progress and shutting down gracefully...")
    is_shutting_down = True
    save_progress()
    print("Progress saved. You can resume later by running the script again.")
    sys.exit(0)

async def retry_with_backoff(func, *args, **kwargs):
    """Execute a function with exponential backoff retry logic."""
    attempt = 0
    last_exception = None
    
    while attempt < MAX_RETRIES:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            attempt += 1
            last_exception = e
            
            if attempt == MAX_RETRIES:
                print(f"Max retries ({MAX_RETRIES}) reached. Giving up.")
                break
                
            # Calculate backoff with jitter
            delay = min(MAX_RETRY_DELAY, BASE_RETRY_DELAY * (2 ** (attempt - 1)))
            # Add jitter
            jitter = random.uniform(-JITTER_FACTOR * delay, JITTER_FACTOR * delay)
            delay = delay + jitter
                
            print(f"Attempt {attempt} failed with error: {str(e)}")
            print(f"Retrying in {delay:.2f} seconds...")
            
            await asyncio.sleep(delay)
    
    # If we get here, all retries failed
    raise last_exception

async def brave_search(query: str, count: int = 3) -> List[Dict[str, Any]]:
    """Perform a web search using Brave Search API.
    
    Note: For content moderation, this is less directly applicable but 
    maintained for potential context lookup of referenced terms or usernames."""
    if not BRAVE_SEARCH_API_KEY:
        print("No Brave Search API key found, skipping search")
        return []
        
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_SEARCH_API_KEY
    }
    
    params = {
        "q": query,
        "count": count
    }
    
    async def _do_search():
        async with aiohttp.ClientSession() as session:
            async with session.get(BRAVE_SEARCH_ENDPOINT, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("web", {}).get("results", [])
                elif response.status == 429:  # Rate limited
                    raise Exception(f"Rate limited by Brave Search API (429). Try again later.")
                else:
                    raise Exception(f"Search failed with status {response.status}")
    
    try:
        return await retry_with_backoff(_do_search)
    except Exception as e:
        print(f"All retries for Brave Search failed: {e}")
        return []

async def check_post_acceptability(post: str) -> Tuple[bool, str]:
    """
    Check if a post is acceptable using DeepSeek.
    Returns a tuple of (is_acceptable, explanation)
    """
    # Load the system prompt from file
    system_prompt = load_system_prompt()
    
    async def _make_api_call():
        response = client.chat.completions.create(
            model=MODEL_TO_USE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Post to evaluate:\n\"{post}\"\n\nIs this post acceptable or does it contain extreme profanity or nudity references? Return your answer as a JSON object with 'acceptable' (boolean) and 'explanation' (string) fields."}
            ],
            max_tokens=1000,
            temperature=0.0,
            top_p=0.95
        )
        return response
    
    # Create the message for DeepSeek (using OpenAI compatible format)
    try:
        # Use retry mechanism for API calls
        response = await retry_with_backoff(_make_api_call)
        
        # Extract response content
        content = response.choices[0].message.content
        
        # Try to parse JSON from the response
        try:
            # Look for JSON in the response - LLMs might wrap it in code blocks
            import re
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If not in code blocks, try to extract the entire JSON object
                json_match = re.search(r'({.*})', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = content
            
            result = json.loads(json_str)
            return (result.get("acceptable", False), result.get("explanation", "No explanation provided"))
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from response: {content}")
            # Default to flagging if we can't parse the response
            return (False, "Error parsing response - flagging as inappropriate by default")
            
    except Exception as e:
        print(f"All retries failed for DeepSeek API: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        # Default to flagging if the API call fails
        return (False, f"API Error: {e} - flagging as inappropriate by default")

def merge_results_with_logs():
    """Merge existing logs with current progress."""
    if not os.path.exists(LOG_FILE):
        return []
    
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            existing_log = json.load(f)
            
        # Convert to dict for easier lookup
        existing_dict = {item["post"]: item for item in existing_log}
        
        # Add completed posts from current progress
        for post_data in current_progress["completed_posts"]:
            existing_dict[post_data["post"]] = post_data
        
        # Convert back to list
        return list(existing_dict.values())
    except Exception as e:
        print(f"Error merging results: {e}")
        return current_progress["completed_posts"]

async def process_posts(batch_size: int = 5):
    """Process posts in batches and sort them into acceptable and flagged files."""
    global current_progress, is_shutting_down
    
    # First, check if we need to resume from previous progress
    all_posts = []
    if load_progress() and current_progress["pending_posts"]:
        # Resume from previous progress
        all_posts = current_progress["pending_posts"]
        print(f"Resuming with {len(all_posts)} pending posts.")
        # Debug: Print what's in the pending posts list
        print(f"Pending posts to process: {[post[:30] + '...' if len(post) > 30 else post for post in all_posts]}")
    else:
        # Start fresh
        try:
            with open(INPUT_FILE, "r", encoding="utf-8") as f:
                all_posts = f.read().split("---\n")
                # Clean up empty posts and whitespace
                all_posts = [post.strip() for post in all_posts if post.strip()]
                
            # Debug: Print what was loaded from the file
            print(f"Posts loaded from file: {len(all_posts)}")
            print(f"Posts after filtering empties: {len(all_posts)}")
            print(f"First few posts: {[post[:30] + '...' if len(post) > 30 else post for post in all_posts[:3]]}")
            
            # Initialize progress tracking
            current_progress = {
                "completed_posts": [],
                "pending_posts": all_posts.copy(),  # Make explicit copy to avoid reference issues
                "total_posts": len(all_posts),
                "last_processed_index": -1
            }
            save_progress()
        except Exception as e:
            print(f"Error reading input file: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return
    
    print(f"Found {len(all_posts)} posts to process")
    
    # Create output files if they don't exist
    if not os.path.exists(ACCEPTABLE_FILE):
        open(ACCEPTABLE_FILE, "w", encoding="utf-8").close()
    if not os.path.exists(FLAGGED_FILE):
        open(FLAGGED_FILE, "w", encoding="utf-8").close()
    
    # Process posts in batches
    total_processed = len(current_progress["completed_posts"])
    processing_log = merge_results_with_logs()
    
    # Debug: Check what's in the processing log at the start
    print(f"Processing log entries at start: {len(processing_log)}")
    
    # Use a copy of the posts list for iteration to avoid modification issues
    pending_posts = all_posts.copy()
    
    # Process pending posts one by one (allows for cleaner tracking)
    for post in pending_posts:
        if is_shutting_down:
            print("Shutdown requested, stopping processing.")
            break
            
        try:
            # Check if the post was already processed
            already_processed = False
            for processed in processing_log:
                if processed["post"] == post:
                    already_processed = True
                    if processed["acceptable"]:
                        # Ensure it's in the acceptable file
                        with open(ACCEPTABLE_FILE, "r", encoding="utf-8") as f:
                            acceptable_content = f.read()
                        if post not in acceptable_content:
                            with open(ACCEPTABLE_FILE, "a", encoding="utf-8") as f:
                                f.write(f"{post}\n---\n")
                    else:
                        # Ensure it's in the flagged file
                        with open(FLAGGED_FILE, "r", encoding="utf-8") as f:
                            flagged_content = f.read()
                        if post not in flagged_content:
                            with open(FLAGGED_FILE, "a", encoding="utf-8") as f:
                                f.write(f"{post}\n---\n")
                    break
            
            if already_processed:
                print(f"Skipped already processed post: {post[:30]}...")
                # Update progress tracking
                if post in current_progress["pending_posts"]:
                    current_progress["pending_posts"].remove(post)
                    print(f"Removed post from pending. Pending count: {len(current_progress['pending_posts'])}")
                total_processed += 1
                save_progress()
                continue
            
            # Process this post
            is_acceptable, explanation = await check_post_acceptability(post)
            
            # Write result to appropriate file
            if is_acceptable:
                with open(ACCEPTABLE_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{post}\n---\n")
            else:
                with open(FLAGGED_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{post}\n---\n")
            
            # Create result record
            post_result = {
                "post": post,
                "acceptable": is_acceptable,
                "explanation": explanation
            }
            
            # Add to overall log
            processing_log.append(post_result)
            
            # Update progress tracking
            current_progress["completed_posts"].append(post_result)
            
            # Critical fix: Make sure the post is actually removed from pending
            if post in current_progress["pending_posts"]:
                current_progress["pending_posts"].remove(post)
                print(f"Post removed from pending. Pending count: {len(current_progress['pending_posts'])}")
            else:
                print(f"WARNING: Post not found in pending list: {post[:30]}...")
                
            current_progress["last_processed_index"] = total_processed
            
            # Update individual progress
            total_processed += 1
            print(f"Processed {total_processed}/{current_progress['total_posts']} posts")
            
            # Debug: check pending count after processing
            print(f"After processing, pending: {len(current_progress['pending_posts'])}, completed: {len(current_progress['completed_posts'])}")
            
            # Save progress after each post to allow for resuming
            save_progress()
            
            # Save log after each post
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(processing_log, f, indent=2)
            
            # Brief pause to avoid rate limiting
            await asyncio.sleep(0.5)
            
        except Exception as e:
            print(f"Error processing post: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            # Save progress so we can resume
            save_progress()
            # Brief pause before trying the next post
            await asyncio.sleep(1)
    
    if not is_shutting_down:
        # Final check to ensure all posts were processed
        if len(current_progress["pending_posts"]) > 0:
            print(f"WARNING: Still have {len(current_progress['pending_posts'])} pending posts after processing.")
            print(f"Pending posts: {[post[:30] + '...' for post in current_progress['pending_posts']]}")
        
        print(f"Processing complete! Results saved to {ACCEPTABLE_FILE} and {FLAGGED_FILE}")
        print(f"Processing log saved to {LOG_FILE}")
        
        # Clear progress file when complete
        if os.path.exists(PROGRESS_FILE):
            try:
                os.remove(PROGRESS_FILE)
                print(f"Progress file removed (processing complete)")
            except Exception as e:
                print(f"Error removing progress file: {e}")
        
        # Print a summary of the flagged posts
        flagged_count = len([item for item in processing_log if not item["acceptable"]])
        print(f"\nSummary: {flagged_count} posts were flagged as inappropriate out of {current_progress['total_posts']} total posts.")
        print("Some example flagged posts:")
        flagged_posts = [item for item in processing_log if not item["acceptable"]]
        for item in flagged_posts[:5]:  # Show up to 5 examples
            preview = item["post"][:50] + "..." if len(item["post"]) > 50 else item["post"]
            print(f"- {preview}")
            print(f"  Reason: {item['explanation'][:100]}...")
    else:
        print("Processing interrupted. Progress saved.")

async def main():
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Terminal close
    
    # Ensure input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} not found. Please create it with your posts separated by '---'")
        return
    
    # Check for API keys
    if not DEEPSEEK_API_KEY:
        print("DEEPSEEK_API_KEY not found. Please set it in your .env file")
        return
    
    if not BRAVE_SEARCH_API_KEY:
        print("BRAVE_SEARCH_API_KEY not found. Please set it in your .env file")
        print("Continuing without web search capability")
    
    # Print selected model
    print(f"Using DeepSeek model: {MODEL_TO_USE} - {DEEPSEEK_MODELS.get(MODEL_TO_USE, 'Unknown model')}")
    
    # Process the posts
    await process_posts()

if __name__ == "__main__":
    asyncio.run(main())