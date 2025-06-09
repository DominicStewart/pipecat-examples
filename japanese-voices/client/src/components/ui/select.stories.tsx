/** @format */

import {
	Select,
	SelectContent,
	SelectGuide,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";

// Simple story component for component documentation
export const SelectDefault = ({
	guide = "",
	size = "default" as "default" | "sm" | "lg",
}: {
	guide?: string;
	size?: "default" | "sm" | "lg";
}) => (
	<Select>
		<SelectTrigger size={size}>
			{guide && <SelectGuide>{guide}</SelectGuide>}
			<SelectValue placeholder="Please select" />
		</SelectTrigger>
		<SelectContent>
			<SelectItem value="item-1">Select Item 1</SelectItem>
			<SelectItem value="item-2">Select Item 2</SelectItem>
			<SelectItem value="item-3">Select Item 3</SelectItem>
		</SelectContent>
	</Select>
);
