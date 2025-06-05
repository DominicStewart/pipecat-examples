/** @format */

"use client";

import dynamic from "next/dynamic";

// Dynamically import the widget to avoid SSR issues
const Widget = dynamic(
	() => import("@/widget").then((mod) => ({ default: mod.Widget })),
	{
		ssr: false,
	}
);

export default function Home() {
	const handleConnect = async () => {
		// This should connect to your bot server
		// For now, this is a placeholder - you'll need to implement the actual connection logic
		// that calls your server to create a Daily room and get the token
		const response = await fetch("/api/connect", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
		});

		if (!response.ok) {
			throw new Error("Failed to connect");
		}

		return response;
	};

	return (
		<div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center p-4">
			<div className="text-center">
				<h1 className="text-4xl md:text-6xl font-bold text-gray-900 dark:text-white mb-4">
					Japanese Voice Bot Test
				</h1>
				<p className="text-lg md:text-xl text-gray-600 dark:text-gray-300 mb-8">
					Click the button below to start speaking with our AI assistant
				</p>
			</div>

			{/* Widget component */}
			<Widget
				onConnect={handleConnect}
				collapsedButtonText="Speak with Japanese AI Assistant"
				enableTextInput={false}
				showTranscription={true}
			/>
		</div>
	);
}
