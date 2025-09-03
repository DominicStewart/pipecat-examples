/** @format */

"use client";

import { useState } from "react";
import { ConsoleTemplate, ThemeProvider } from "@pipecat-ai/voice-ui-kit";
import { Phone, Sparkles, ArrowRight, CheckCircle } from "lucide-react";

export default function Home() {
	const [showConsole, setShowConsole] = useState(false);
	const [phoneNumber, setPhoneNumber] = useState("");
	const [callerId, setCallerId] = useState("");
	const [isLoading, setIsLoading] = useState(false);

	const handleStartCall = async (e: React.FormEvent) => {
		e.preventDefault();
		if (!phoneNumber.trim()) return;

		setIsLoading(true);

		// Optional: You could call your API here first to initiate the call
		// with the specific phone number, then show the console

		// For now, just show the console after a brief delay
		setTimeout(() => {
			setShowConsole(true);
			setIsLoading(false);
		}, 1000);
	};

	if (showConsole) {
		return (
			<ThemeProvider>
				<div className="w-full h-dvh bg-background relative">
					<button
						onClick={() => setShowConsole(false)}
						className="absolute top-4 left-4 z-50 px-4 py-2 bg-white/10 backdrop-blur-sm text-white rounded-lg hover:bg-white/20 transition-all duration-200 flex items-center gap-2"
					>
						← Back to Setup
					</button>
					<ConsoleTemplate
						connectParams={{
							endpoint: "/api/connect",
						}}
						transportType="daily"
					/>
				</div>
			</ThemeProvider>
		);
	}

	return (
		<ThemeProvider>
			<div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 dark:from-gray-900 dark:via-blue-900 dark:to-purple-900">
				{/* Background decoration */}
				<div className="absolute inset-0 overflow-hidden">
					<div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-300 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob"></div>
					<div className="absolute -bottom-40 -left-40 w-80 h-80 bg-yellow-300 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-2000"></div>
					<div className="absolute top-40 left-40 w-80 h-80 bg-pink-300 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-4000"></div>
				</div>

				<div className="relative z-10 flex items-center justify-center min-h-screen p-6">
					<div className="w-full max-w-md">
						{/* Header */}
						<div className="text-center mb-8">
							<div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl mb-6 shadow-lg">
								<Phone className="w-8 h-8 text-white" />
							</div>
							<h1 className="text-4xl font-bold bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-300 bg-clip-text text-transparent mb-3">
								Voice Bot Caller
							</h1>
							<p className="text-gray-600 dark:text-gray-400 text-lg flex items-center justify-center gap-2">
								<Sparkles className="w-5 h-5 text-purple-500" />
								Make a call with AI voice assistance
							</p>
						</div>

						{/* Main Form Card */}
						<div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl rounded-3xl shadow-2xl border border-white/20 p-8 mb-6">
							<form onSubmit={handleStartCall} className="space-y-6">
								{/* Phone Number Input */}
								<div className="space-y-2">
									<label
										htmlFor="phoneNumber"
										className="block text-sm font-semibold text-gray-700 dark:text-gray-300"
									>
										Phone Number *
									</label>
									<input
										type="tel"
										id="phoneNumber"
										value={phoneNumber}
										onChange={(e) => setPhoneNumber(e.target.value)}
										placeholder="+1234567890"
										className="w-full px-4 py-3 bg-white/50 dark:bg-gray-700/50 border border-gray-200 dark:border-gray-600 rounded-xl text-gray-900 dark:text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
										required
									/>
									<p className="text-xs text-gray-500 dark:text-gray-400">
										Include country code (e.g., +1 for US)
									</p>
								</div>

								{/* Caller ID Input */}
								<div className="space-y-2">
									<label
										htmlFor="callerId"
										className="block text-sm font-semibold text-gray-700 dark:text-gray-300"
									>
										Caller ID (Optional)
									</label>
									<input
										type="tel"
										id="callerId"
										value={callerId}
										onChange={(e) => setCallerId(e.target.value)}
										placeholder="+1234567890"
										className="w-full px-4 py-3 bg-white/50 dark:bg-gray-700/50 border border-gray-200 dark:border-gray-600 rounded-xl text-gray-900 dark:text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
									/>
									<p className="text-xs text-gray-500 dark:text-gray-400">
										Number to display to the recipient
									</p>
								</div>

								{/* Start Call Button */}
								<button
									type="submit"
									disabled={!phoneNumber.trim() || isLoading}
									className="w-full group relative px-6 py-4 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-[1.02] disabled:hover:scale-100"
								>
									<div className="flex items-center justify-center gap-3">
										{isLoading ? (
											<>
												<div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
												Connecting...
											</>
										) : (
											<>
												<Phone className="w-5 h-5" />
												Proceed to Console
												<ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform duration-200" />
											</>
										)}
									</div>
								</button>
							</form>
						</div>

						{/* How it works section */}
						<div className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-xl rounded-2xl border border-white/20 p-6">
							<h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
								<CheckCircle className="w-5 h-5 text-green-500" />
								How it works:
							</h3>
							<ol className="space-y-3 text-sm text-gray-600 dark:text-gray-400">
								<li className="flex items-start gap-3">
									<span className="flex-shrink-0 w-6 h-6 bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full text-xs font-semibold flex items-center justify-center">
										1
									</span>
									Enter the phone number you want to call
								</li>
								<li className="flex items-start gap-3">
									<span className="flex-shrink-0 w-6 h-6 bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full text-xs font-semibold flex items-center justify-center">
										2
									</span>
									Optionally set a caller ID
								</li>
								<li className="flex items-start gap-3">
									<span className="flex-shrink-0 w-6 h-6 bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full text-xs font-semibold flex items-center justify-center">
										3
									</span>
									Click Start Call to begin the voice session
								</li>
								<li className="flex items-start gap-3">
									<span className="flex-shrink-0 w-6 h-6 bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full text-xs font-semibold flex items-center justify-center">
										4
									</span>
									The bot will call the number and handle voicemail or human
									detection
								</li>
							</ol>
						</div>
					</div>
				</div>
			</div>

			<style jsx>{`
				@keyframes blob {
					0% {
						transform: translate(0px, 0px) scale(1);
					}
					33% {
						transform: translate(30px, -50px) scale(1.1);
					}
					66% {
						transform: translate(-20px, 20px) scale(0.9);
					}
					100% {
						transform: translate(0px, 0px) scale(1);
					}
				}
				.animate-blob {
					animation: blob 7s infinite;
				}
				.animation-delay-2000 {
					animation-delay: 2s;
				}
				.animation-delay-4000 {
					animation-delay: 4s;
				}
			`}</style>
		</ThemeProvider>
	);
}
