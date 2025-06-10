/** @format */

import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
	try {
		// Log the request for debugging
		console.log("Received bot connection request:", request.method);
		console.log("Environment variables:", {
			FASTAPI_SERVER_URL: process.env.FASTAPI_SERVER_URL,
			PIPECAT_CLOUD_API_KEY: process.env.PIPECAT_CLOUD_API_KEY
				? "[SET]"
				: "[NOT SET]",
		});

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
		console.log("Using server URL:", serverUrl);
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
		// Call the FastAPI server's /start endpoint
		console.log("Making request to:", `${serverUrl}/start`);
		console.log("Request headers:", headers_content);
		console.log("Request body:", JSON.stringify(body_content));

		const response = await fetch(`${serverUrl}/start`, {
			method: "POST",
			headers: headers_content,
			body: JSON.stringify(body_content),
		});

		console.log("Response status:", response.status);
		console.log(
			"Response headers:",
			Object.fromEntries(response.headers.entries())
		);

		if (!response.ok) {
			const errorText = await response.text();
			console.error("Failed to start bot via FastAPI server:", errorText);
			console.error("Response status:", response.status);
			return NextResponse.json(
				{
					error: "Failed to start bot server",
					details: errorText,
					status: response.status,
					serverUrl: serverUrl,
				},
				{ status: 500 }
			);
		}

		const data = await response.json();
		console.log("Bot started successfully:", data);
		console.log("Data keys:", Object.keys(data));
		console.log("dailyRoom type:", typeof data.dailyRoom);
		console.log("dailyRoom value:", data.dailyRoom);
		console.log("dailyToken type:", typeof data.dailyToken);
		console.log("dailyToken value:", data.dailyToken);

		// Handle both property name formats (room_url/token and dailyRoom/dailyToken)
		const roomUrl = data.room_url || data.dailyRoom;
		const token = data.token || data.dailyToken;

		// Validate the response data
		if (!roomUrl || typeof roomUrl !== "string") {
			console.error("Invalid room URL in response:", roomUrl);
			return NextResponse.json(
				{
					error: "Invalid response from bot server",
					details: `room URL is ${typeof roomUrl}: ${roomUrl}`,
					fullResponse: data,
				},
				{ status: 500 }
			);
		}

		if (!token || typeof token !== "string") {
			console.error("Invalid token in response:", token);
			return NextResponse.json(
				{
					error: "Invalid response from bot server",
					details: `token is ${typeof token}: ${token}`,
					fullResponse: data,
				},
				{ status: 500 }
			);
		}

		// Return the response in the format expected by the widget
		return NextResponse.json({
			url: roomUrl, // Daily.co expects 'url', not 'room_url'
			token: token,
			// config: [
			// 	{
			// 		service: "tts",
			// 		options: [{ name: "voice", value: "alloy" }],
			// 	},
			// ],
		});
	} catch (error) {
		console.error("Error in bot connection endpoint:", error);
		console.error("Error details:", {
			message: error instanceof Error ? error.message : String(error),
			stack: error instanceof Error ? error.stack : undefined,
			serverUrl: process.env.FASTAPI_SERVER_URL,
		});
		return NextResponse.json(
			{
				error: "Failed to connect to bot server",
				details: error instanceof Error ? error.message : String(error),
			},
			{ status: 500 }
		);
	}
}
