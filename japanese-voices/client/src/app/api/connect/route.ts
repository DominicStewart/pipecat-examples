/** @format */

import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
	try {
		// Log the request for debugging
		console.log("Received bot connection request:", request.method);
		
		// Safely parse JSON with fallback for empty body
		let requestBody: { MY_CUSTOM_DATA?: unknown } = {};
		try {
			const text = await request.text();
			if (text) {
				requestBody = JSON.parse(text);
			}
		} catch {
			console.log("No JSON body provided, using empty object");
		}
		
		const { MY_CUSTOM_DATA } = requestBody;

		// Get the FastAPI server URL from environment (use 127.0.0.1 to force IPv4)
		const serverUrl = process.env.FASTAPI_SERVER_URL || "http://127.0.0.1:7860";
		let headers_content;
		console.log("Calling FastAPI server to start bot...");
		if (serverUrl === "http://127.0.0.1:7860") {
			headers_content = {
				"Content-Type": "application/json",
			};
		} else {
			headers_content = {
				Authorization: `Bearer ${process.env.PIPECAT_CLOUD_API_KEY}`,
				"Content-Type": "application/json",
			};
		}
		let body_content;
		if (serverUrl === "http://127.0.0.1:7860") {
			body_content = {};
		} else {
			body_content = {
				// Create Daily room
				createDailyRoom: true,
				// Optionally set Daily room properties
				dailyRoomProperties: { start_video_off: true },
				// Optionally pass custom data to the bot
				body: { MY_CUSTOM_DATA },
			};
		}
		// Call the FastAPI server's /connect endpoint
		const response = await fetch(`${serverUrl}/start`, {
			method: "POST",
			headers: headers_content,
			body: JSON.stringify(body_content),
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
			// config: [
			// 	{
			// 		service: "tts",
			// 		options: [{ name: "voice", value: "alloy" }],
			// 	},
			// ],
		});
	} catch (error) {
		console.error("Error in bot connection endpoint:", error);
		return NextResponse.json(
			{ error: "Failed to connect to bot server" },
			{ status: 500 }
		);
	}
}
