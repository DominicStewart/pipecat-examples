import type { Story, StoryDefault } from "@ladle/react";
import { LayoutSection } from "./layout";

export default {
  title: "Primitives",
} satisfies StoryDefault;

export const LayoutSectionDefault: Story = () => (
  <LayoutSection key="section-1">
    <div>
      <h1>Section 1</h1>
    </div>
  </LayoutSection>
);
