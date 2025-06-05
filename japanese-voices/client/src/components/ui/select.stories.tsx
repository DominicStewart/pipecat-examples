import {
  Select,
  SelectContent,
  SelectGuide,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { Story, StoryDefault } from "@ladle/react";

export default {
  title: "Primitives",
} satisfies StoryDefault;

export const SelectDefault: Story<{
  guide?: string;
  size: "default" | "sm";
}> = ({ guide, size }) => (
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

SelectDefault.args = {
  guide: "",
};
SelectDefault.argTypes = {
  size: {
    options: ["default", "sm", "lg"],
    control: { type: "radio" },
    defaultValue: "default",
  },
};

SelectDefault.storyName = "Select";
