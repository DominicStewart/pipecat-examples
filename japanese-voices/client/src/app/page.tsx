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
		<div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 flex flex-col items-center p-4 pb-20 overflow-x-hidden">
			{/* Content area that can scroll if needed */}
			<div className="flex-1 flex flex-col items-center justify-center gap-8 max-w-4xl w-full">
				<div className="text-center px-4">
					<h1 className="text-3xl sm:text-4xl md:text-6xl font-bold text-gray-900 dark:text-white mb-4">
						Japanese Voice Bot Test
					</h1>
					<p className="text-base sm:text-lg md:text-xl text-gray-600 dark:text-gray-300">
						Click the button below to start speaking with our AI assistant
					</p>
				</div>
			</div>

			{/* Widget container positioned at bottom with space to expand upward */}
			<div className="fixed bottom-4 left-1/2 transform -translate-x-1/2 w-full max-w-xs sm:max-w-sm px-4 z-50 flex flex-col-reverse items-center">
				<Widget
					onConnect={handleConnect}
					collapsedButtonText="Speak with Japanese AI Assistant"
					enableTextInput={false}
					showTranscription={true}
				/>
			</div>
		</div>
	);
}
