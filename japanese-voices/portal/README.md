# Japanese Voice Bot Portal

A Next.js web portal for connecting to the Japanese voice bot powered by Pipecat.

## Features

- **Simple Interface**: Clean, centered portal with "Japanese Voice Bot Test" title
- **Voice Widget**: Interactive voice interface for speaking with the Japanese bot
- **TypeScript Safe**: Fully typed codebase for better development experience
- **Vercel Ready**: Configured for easy deployment to Vercel

## How It Works

1. **Portal Interface**: The main page displays a centered title and voice widget
2. **Connection API**: `/api/connect` endpoint creates Daily rooms and manages bot connections
3. **Voice Widget**: Uses Pipecat React components for seamless voice interaction
4. **Bot Integration**: Automatically connects to the Japanese voice bot server

## Setup

### Prerequisites

- Node.js 18.18.0 (managed automatically with `.nvmrc`)
- Daily API key
- Bot server dependencies (see `../server/requirements.txt`)

### Environment Variables

Create a `.env.local` file with:

```bash
DAILY_API_KEY=your_daily_api_key_here
DAILY_API_URL=https://api.daily.co/v1
```

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

The portal will be available at `http://localhost:3000`.

## Usage

1. Open the portal in your browser
2. Click the voice widget to expand it
3. Click "Connect" to start a new session
4. The system will:
   - Create a new Daily room
   - Generate a participant token
   - Return connection details to the widget
   - Allow you to speak with the Japanese voice bot

## Manual Bot Server Setup

Currently, you need to manually start the bot server after creating a session:

1. Use the portal to create a session (this creates the Daily room)
2. Check the browser console for the room URL and token
3. Start the bot server with:

```bash
cd ../server
DAILY_ROOM_URL=<room_url> DAILY_ROOM_TOKEN=<token> python bot.py
```

## Architecture

- **Frontend**: Next.js 14 with TypeScript and Tailwind CSS
- **Voice Components**: Pipecat React widgets (`@pipecat-ai/client-react`)
- **Transport**: Daily.co WebRTC transport (`@pipecat-ai/daily-transport`)
- **API**: Next.js API routes for room management
- **Bot**: Python server with Pipecat framework

## Files Structure

```
portal/
├── src/
│   ├── app/
│   │   ├── page.tsx              # Main portal page
│   │   └── api/connect/route.ts  # Connection API endpoint
│   ├── widget/                   # Voice widget components
│   └── lib/utils.ts             # Utility functions
├── start_bot.py                 # Bot server starter script
└── .env.local                   # Environment configuration
```

## Deployment

This portal is configured for Vercel deployment:

1. Connect your repository to Vercel
2. Set environment variables in Vercel dashboard
3. Deploy automatically on push to main branch

## Future Enhancements

- [ ] Automatic bot server process management
- [ ] Session persistence and management
- [ ] Multiple voice options
- [ ] Chat transcription display
- [ ] Room cleanup and management
