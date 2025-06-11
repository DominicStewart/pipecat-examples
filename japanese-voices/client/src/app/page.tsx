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
	// Function that calls
	return (
		<>
			{/* Main page content */}
			<main className="relative min-h-screen w-full bg-gradient-to-br from-indigo-500/10 via-purple-500/10 to-pink-500/10 overflow-hidden">
				{/* Enhanced decorative elements */}
				<div className="absolute inset-0 overflow-hidden pointer-events-none">
					<div className="absolute -top-40 -right-40 w-80 h-80 rounded-full bg-pink-500/20 blur-3xl animate-pulse" />
					<div className="absolute top-1/4 -left-20 w-60 h-60 rounded-full bg-indigo-500/20 blur-3xl animate-pulse delay-1000" />
					<div className="absolute bottom-0 left-1/3 w-72 h-72 rounded-full bg-purple-500/20 blur-3xl animate-pulse delay-2000" />

					{/* Additional floating elements */}
					<div className="absolute top-1/2 right-1/4 w-32 h-32 rounded-full bg-cyan-400/10 blur-2xl animate-bounce" />
					<div className="absolute bottom-1/4 left-1/4 w-24 h-24 rounded-full bg-rose-400/10 blur-xl animate-bounce delay-500" />
				</div>

				{/* Content with mobile-safe padding */}
				<div className="relative z-10 container mx-auto px-4 py-8 pb-32 md:pb-16">
					{/* Hero Section */}
					<div className="text-center mb-12 pt-8">
						<h1 className="text-4xl md:text-6xl lg:text-7xl font-bold mb-6 gradient-text leading-tight">
							Japanese Voice
							<br />
							<span className="text-3xl md:text-5xl lg:text-6xl">
								(TTS) Demo
							</span>
						</h1>

						<div className="max-w-2xl mx-auto">
							<p className="text-lg md:text-xl text-gray-700 dark:text-gray-300 mb-8 leading-relaxed">
								Experience the quality of various Japanese Text-To-Speech
								services through our interactive AI assistant. Assistant will
								start with the male OpenAI TTS voice. You can switch voices or
								providers at any time.
							</p>
						</div>
					</div>

					{/* Instructions Card */}
					<div className="max-w-4xl mx-auto mb-12">
						<div className="bg-white/10 dark:bg-black/20 backdrop-blur-lg rounded-2xl p-6 md:p-8 border border-white/20 dark:border-gray-700/30 shadow-2xl">
							<h2 className="text-2xl md:text-3xl font-semibold mb-6 text-center gradient-text">
								Voice Commands
							</h2>

							<div className="grid md:grid-cols-2 gap-6 mb-8">
								<div className="space-y-4">
									<div className="bg-gradient-to-r from-indigo-500/20 to-purple-500/20 rounded-xl p-4 border border-indigo-300/30">
										<h3 className="font-semibold text-indigo-300 dark:text-indigo-200 mb-2 flex items-center">
											<span className="w-2 h-2 bg-indigo-400 rounded-full mr-3"></span>
											Switch TTS Provider
										</h3>
										<p className="text-sm text-gray-600 dark:text-gray-300">
											&ldquo;I would like to switch to Cartesia / Elevenlabs /
											OpenAI / Azure / Google&rdquo;
										</p>
									</div>

									<div className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-xl p-4 border border-purple-300/30">
										<h3 className="font-semibold text-purple-300 dark:text-purple-200 mb-2 flex items-center">
											<span className="w-2 h-2 bg-purple-400 rounded-full mr-3"></span>
											Change Voice Gender
										</h3>
										<p className="text-sm text-gray-600 dark:text-gray-300">
											&ldquo;Switch to a male / female voice&rdquo;
										</p>
									</div>
								</div>

								<div className="bg-gradient-to-r from-cyan-500/20 to-blue-500/20 rounded-xl p-4 border border-cyan-300/30">
									<h3 className="font-semibold text-cyan-300 dark:text-cyan-200 mb-2 flex items-center">
										<span className="w-2 h-2 bg-cyan-400 rounded-full mr-3"></span>
										Language Support
									</h3>
									<p className="text-sm text-gray-600 dark:text-gray-300">
										Ask in English, get responses in English. Story is read in
										Japanese
									</p>
								</div>
							</div>
						</div>

						{/* Literary Reference */}
						<div className="text-center bg-gradient-to-r from-amber-500/10 to-orange-500/10 rounded-xl p-6 border border-amber-300/20">
							<h3 className="font-semibold text-amber-600 dark:text-amber-300 mb-2">
								📚 Featured Literature
							</h3>
							<p className="text-gray-700 dark:text-gray-300">
								The assistant will read excerpts from{" "}
								<em>&ldquo;Wagahai wa Neko de Aru&rdquo;</em>
								<br className="hidden sm:block" />
								(吾輩は猫である) by Natsume Sōseki
							</p>
						</div>
					</div>
				</div>

				{/* Call to Action */}
				<div className="text-center max-w-md mx-auto">
					<div className="bg-gradient-to-r from-indigo-500/20 to-purple-500/20 rounded-2xl p-6 border border-indigo-300/30 backdrop-blur-sm">
						<h3 className="text-xl font-semibold mb-3 gradient-text">
							Ready to Experience Japanese TTS?
						</h3>
						<p className="text-gray-600 dark:text-gray-300 mb-4">
							Click the assistant button to start your voice conversation
						</p>
						<div className="flex items-center justify-center space-x-2 text-sm text-gray-500 dark:text-gray-400">
							<span>Look for the</span>
							<div className="bg-gray-800 text-white px-3 py-1 rounded-full text-xs">
								Assistant
							</div>
							<span>button ↘️</span>
						</div>
					</div>
				</div>
			</main>

			{/* Fixed positioned widget - will appear in bottom right */}
			<Widget
				onConnect={async () => {
					const response = await fetch("/api/connect", {
						method: "POST",
						headers: {
							"Content-Type": "application/json",
						},
						body: JSON.stringify({
							MY_CUSTOM_DATA: {}, // or whatever data you need
						}),
					});
					console.log("Response from connect endpoint:", response);
					if (!response.ok) {
						throw new Error("Failed to connect to bot");
					}

					const data = await response.json();
					console.log("Data from connect endpoint:", data);
					if (data.error) {
						throw new Error(data.error);
					}

					return new Response(
						JSON.stringify({
							room_url: data.room_url,
							token: data.token,
						}),
						{ status: 200 }
					);
				}}
				collapsedButtonText="Speak with Japanese AI Assistant"
				enableTextInput={false}
				showTranscription={false}
			/>
		</>
	);
}
