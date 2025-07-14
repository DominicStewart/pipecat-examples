/** @format */

import type { Config } from "tailwindcss";

export default {
	content: [
		"./pages/**/*.{js,ts,jsx,tsx,mdx}",
		"./components/**/*.{js,ts,jsx,tsx,mdx}",
		"./app/**/*.{js,ts,jsx,tsx,mdx}",
	],
	theme: {
		extend: {
			fontFamily: {
				sans: ["Geist Variable", "sans-serif"],
				mono: ["Geist Mono Variable", "monospace"],
			},
		},
	},
} satisfies Config;
