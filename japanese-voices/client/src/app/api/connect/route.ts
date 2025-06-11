/** @format */

import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
	try {
		// Log the request for debugging
		console.log("Received bot connection request:", request.method);

		// Parse request body (optional, since your client doesn't send much)
		const body = await request.json().catch(() => ({}));

		// Get the server URL from environment
		const serverUrl =
			process.env.FASTAPI_SERVER_URL ||
			"https://api.pipecat.daily.co/v1/public/jpvoice/start";
		const authKey = process.env.PIPECAT_CLOUD_API_KEY;

		console.log("Using server URL:", serverUrl);
		console.log("authKey:", authKey ? "Found" : "Missing");

		// Prepare headers
		const headers = {
			"Content-Type": "application/json",
			...(authKey && { Authorization: `Bearer ${authKey}` }),
		};

		// Prepare request body
		const requestBody = {
			createDailyRoom: true,
			// Add any additional data from the request if needed
			...body,
		};

		console.log("Calling external API...");

		// Call the external API
		const response = await fetch(serverUrl, {
			method: "POST",
			headers,
			body: JSON.stringify(requestBody),
		});

		if (!response.ok) {
			const errorText = await response.text();
			console.error("Failed to connect to external API:", errorText);
			return NextResponse.json(
				{
					error: "Failed to connect to Pipecat",
					details: errorText,
				},
				{ status: response.status }
			);
		}

		const data = await response.json();
		console.log("External API response:", data);

		// Check for API-level errors
		if (data.error) {
			console.error("API returned error:", data.error);
			return NextResponse.json({ error: data.error }, { status: 400 });
		}

		// Return the response in the format expected by the widget
		return NextResponse.json({
			room_url: data.dailyRoom,
			token: data.dailyToken,
		});
	} catch (error) {
		console.error("Error in connect endpoint:", error);
		return NextResponse.json(
			{ error: "Failed to connect to bot server" },
			{ status: 500 }
		);
	}
}
