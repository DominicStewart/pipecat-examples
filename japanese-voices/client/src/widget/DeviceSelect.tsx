/** @format */

"use client";

import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { MicIcon } from "@/icons";
import {
	useRTVIClient,
	useRTVIClientMediaDevices,
} from "@pipecat-ai/client-react";
import { memo, useEffect } from "react";

const DeviceSelect: React.FC = () => {
	const client = useRTVIClient();
	const { availableMics, selectedMic, updateMic } = useRTVIClientMediaDevices();

	useEffect(() => {
		if (!client) return;

		if (["idle", "disconnected"].includes(client.state)) {
			client.initDevices();
		}
	}, [client]);

	return (
		<div className="flex flex-col gap-2">
			<div className="flex flex-row gap-2 md:gap-3 items-center">
				<MicIcon className="size-6 text-muted-foreground ml-2" />
				<Select
					onValueChange={(value) => updateMic(value)}
					value={selectedMic?.deviceId || undefined}
				>
					<SelectTrigger size="lg" className="w-full font-sans text-sm">
						<SelectValue
							placeholder={
								selectedMic && selectedMic.label
									? selectedMic.label
									: "Loading devices..."
							}
						/>
					</SelectTrigger>
					<SelectContent>
						{availableMics.map((mic) => (
							<SelectItem key={mic.deviceId} value={mic.deviceId}>
								{mic.label}
							</SelectItem>
						))}
					</SelectContent>
				</Select>
			</div>
		</div>
	);
};

export default memo(DeviceSelect);
