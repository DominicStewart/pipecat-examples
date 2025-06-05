import { ConsoleLayout } from "@/components/layouts";
import type { Story, StoryDefault } from "@ladle/react";

export default {
  title: "Layouts",
} satisfies StoryDefault;

export const Console: Story = () => <ConsoleLayout />;
