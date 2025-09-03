/** @format */
import { NextResponse } from "next/server";

export async function POST(request: Request) {
	try {
		// Try to parse JSON body, but make it optional
		let body = {};
		try {
			const requestText = await request.text();
			console.log("Raw request body:", requestText);

			if (requestText.trim()) {
				body = JSON.parse(requestText);
			}
		} catch {
			console.log("No valid JSON body, using empty object");
		}

		console.log("Connect request received:", body);

		// Forward to your Python server's /start endpoint
		const response = await fetch("http://127.0.0.1:7860/start", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({
				dialout_settings: {
					phone_number: "+1234567890", // Default for testing
				},
			}),
		});

		console.log("Python server response status:", response.status);
		console.log(
			"Python server response headers:",
			Object.fromEntries(response.headers)
		);

		if (!response.ok) {
			const errorText = await response.text();
			console.error("Python server error response:", errorText);
			throw new Error(
				`Failed to connect to Pipecat: ${response.status} ${response.statusText} - ${errorText}`
			);
		}

		const responseText = await response.text();
		console.log("Raw response from Python server:", responseText);

		let data;
		try {
			data = JSON.parse(responseText);
		} catch (parseError) {
			console.error("Failed to parse JSON response:", parseError);
			console.error("Response was:", responseText);
			throw new Error(`Invalid JSON response from server: ${responseText}`);
		}

		return NextResponse.json({
			room_url: data.room_url,
			token: data.token,
		});
	} catch (error) {
		console.error("Full error object:", error);

		const errorMessage = error instanceof Error ? error.message : String(error);

		return NextResponse.json(
			{ error: `Failed to process connection request: ${errorMessage}` },
			{ status: 500 }
		);
	}
}
