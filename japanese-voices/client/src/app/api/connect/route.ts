/** @format */

import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
	try {
		// Log the request for debugging
		console.log("Received bot connection request:", request.method);

		// Get Daily API key from environment
		const dailyApiKey = process.env.DAILY_API_KEY;
		const dailyApiUrl = process.env.DAILY_API_URL || "https://api.daily.co/v1";

		if (!dailyApiKey) {
			console.error("DAILY_API_KEY environment variable is not set");
			return NextResponse.json(
				{
					error: "Server configuration error: Missing Daily API key",
				},
				{ status: 500 }
			);
		}

		// Create a Daily room
		console.log("Creating Daily room...");
		const roomResponse = await fetch(`${dailyApiUrl}/rooms`, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				Authorization: `Bearer ${dailyApiKey}`,
			},
			body: JSON.stringify({
				properties: {
					max_participants: 10,
					enable_chat: true,
					enable_knocking: false,
					enable_prejoin_ui: false,
					enable_network_ui: false,
					enable_screenshare: false,
					enable_recording: false,
					enable_transcription: false,
					eject_at_room_exp: true,
					exp: Math.floor(Date.now() / 1000) + 60 * 60, // 1 hour from now
				},
			}),
		});

		if (!roomResponse.ok) {
			const errorText = await roomResponse.text();
			console.error("Failed to create Daily room:", errorText);
			return NextResponse.json(
				{
					error: "Failed to create Daily room",
				},
				{ status: 500 }
			);
		}

		const roomData = await roomResponse.json();
		console.log("Daily room created:", roomData.name);

		// Create a participant token for the room
		const tokenResponse = await fetch(`${dailyApiUrl}/meeting-tokens`, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				Authorization: `Bearer ${dailyApiKey}`,
			},
			body: JSON.stringify({
				properties: {
					room_name: roomData.name,
					is_owner: false,
					enable_recording: false,
					enable_transcription: false,
					exp: Math.floor(Date.now() / 1000) + 60 * 60, // 1 hour from now
				},
			}),
		});

		if (!tokenResponse.ok) {
			const errorText = await tokenResponse.text();
			console.error("Failed to create meeting token:", errorText);
			return NextResponse.json(
				{
					error: "Failed to create meeting token",
				},
				{ status: 500 }
			);
		}

		const tokenData = await tokenResponse.json();
		console.log("Meeting token created");

		// TODO: Start the bot server process
		// For now, we'll skip starting the bot server automatically
		// You would typically spawn a subprocess here that runs:
		// python server/bot.py with the room URL and token as environment variables

		console.log("Bot server startup would happen here");
		console.log("Room URL:", roomData.url);
		console.log("You can manually start the bot server with:");
		console.log(
			`DAILY_ROOM_URL=${roomData.url} DAILY_ROOM_TOKEN=${tokenData.token} python server/bot.py`
		);

		return NextResponse.json({
			room_url: roomData.url,
			token: tokenData.token,
			config: [
				{
					service: "tts",
					options: [{ name: "voice", value: "alloy" }],
				},
			],
		});
	} catch (error) {
		console.error("Error in bot connection endpoint:", error);
		return NextResponse.json(
			{ error: "Failed to connect to bot server" },
			{ status: 500 }
		);
	}
}
