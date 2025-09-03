/** @format */

import type { Metadata } from "next";
import "@fontsource-variable/geist";
import "@fontsource-variable/geist-mono";
import "./globals.css";
import "@pipecat-ai/voice-ui-kit/styles.css";

export const metadata: Metadata = {
	title: "Pipecat Voice Bot",
	description: "Voice AI calling interface",
};

export default function RootLayout({
	children,
}: {
	children: React.ReactNode;
}) {
	return (
		<html lang="en">
			<body className="font-sans antialiased">{children}</body>
		</html>
	);
}
