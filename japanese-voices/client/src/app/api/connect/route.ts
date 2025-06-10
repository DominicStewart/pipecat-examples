/** @format */

import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
	try {
		// Log the request for debugging
		console.log("Received bot connection request:", request.method);

		// Get the FastAPI server URL from environment (use 127.0.0.1 to force IPv4)
		const serverUrl = process.env.FASTAPI_SERVER_URL || "http://127.0.0.1:7860";

		console.log("Calling FastAPI server to start bot...");

		// Call the FastAPI server's /connect endpoint
		const response = await fetch(`${serverUrl}/connect`, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
		});

		if (!response.ok) {
			const errorText = await response.text();
			console.error("Failed to start bot via FastAPI server:", errorText);
			return NextResponse.json(
				{
					error: "Failed to start bot server",
					details: errorText,
				},
				{ status: 500 }
			);
		}

		const data = await response.json();
		console.log("Bot started successfully:", data);

		// Return the response in the format expected by the widget
		return NextResponse.json({
			room_url: data.room_url,
			token: data.token,
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
