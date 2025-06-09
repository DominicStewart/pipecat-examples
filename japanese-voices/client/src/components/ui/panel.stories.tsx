import {
  Panel,
  PanelContent,
  PanelHeader,
  PanelTitle,
} from "@/components/ui/panel";

// Simple story components for component documentation

export const PanelHeaderDefault = ({ 
  label = "Hello world",
  variant = "default" as "default" | "inline"
}: {
  label?: string;
  variant?: "default" | "inline";
}) => (
  <Panel>
    <PanelHeader variant={variant}>
      <PanelTitle>{label}</PanelTitle>
    </PanelHeader>
    <PanelContent>My Panel</PanelContent>
  </Panel>
);
