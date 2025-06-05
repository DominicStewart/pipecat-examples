import {
  Panel,
  PanelContent,
  PanelHeader,
  PanelTitle,
} from "@/components/ui/panel";
import type { Story, StoryDefault } from "@ladle/react";

export default {
  title: "Primitives",
} satisfies StoryDefault;

export const PanelHeaderDefault: Story<{
  label: string;
  variant: "default" | "inline";
}> = ({ label, variant }) => (
  <Panel>
    <PanelHeader variant={variant}>
      <PanelTitle>{label}</PanelTitle>
    </PanelHeader>
    <PanelContent>My Panel</PanelContent>
  </Panel>
);

PanelHeaderDefault.args = {
  label: "Hello world",
};
PanelHeaderDefault.argTypes = {
  variant: {
    options: ["default", "inline"],
    control: { type: "radio" },
    defaultValue: "default",
  },
};

PanelHeaderDefault.storyName = "Panel";
