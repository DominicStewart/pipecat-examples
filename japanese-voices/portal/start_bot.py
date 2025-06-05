#!/usr/bin/env python3
"""
Bot server starter script for the Japanese voice bot portal.
This script is called by the portal API to start a bot session.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the server directory to the path
server_dir = Path(__file__).parent.parent / "server"
sys.path.insert(0, str(server_dir))

from bot import run_bot


async def main():
    """Start the bot with the provided room URL and token."""
    room_url = os.getenv("DAILY_ROOM_URL")
    token = os.getenv("DAILY_ROOM_TOKEN")
    daily_api_key = os.getenv("DAILY_API_KEY")
    daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")
    
    if not all([room_url, token, daily_api_key]):
        print("Missing required environment variables:")
        print(f"DAILY_ROOM_URL: {'✓' if room_url else '✗'}")
        print(f"DAILY_ROOM_TOKEN: {'✓' if token else '✗'}")
        print(f"DAILY_API_KEY: {'✓' if daily_api_key else '✗'}")
        sys.exit(1)
    
    print("Starting Japanese voice bot...")
    print(f"Room URL: {room_url}")
    print(f"API URL: {daily_api_url}")
    
    try:
        await run_bot(room_url, token, daily_api_key, daily_api_url)
    except Exception as e:
        print(f"Error running bot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
