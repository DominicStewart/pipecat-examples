import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { VoiceVisualizer } from "@/components/widget/VoiceVisualizer";
import { ChevronDownIcon, ChevronUpIcon, PhoneCallIcon } from "@/icons";
import { cn } from "@/lib/utils";
import { EndSessionButton } from "./EndSessionButton";
import type { WidgetState } from "./Widget";

interface TogglePanelProps {
  onEndSession: () => void;
  onOpen: () => void;
  onClose?: () => void;
  isOpen?: boolean;
  state?: WidgetState;
  collapsedButtonText?: string;
}

const TogglePanelSessionActive = ({
  onEndSession,
  onOpen,
  state,
}: {
  onEndSession: () => void;
  onOpen: () => void;
  state?: WidgetState;
}) => {
  return (
    <div className="flex flex-row gap-1 md:gap-2">
      <div className="flex flex-1 justify-center items-center mx-2 md:md-3">
        <VoiceVisualizer
          participantType="bot"
          barWidth={4}
          barMaxHeight={38}
          barGap={4}
          barCount={8}
          barColor="#10b981"
        />
      </div>
      <EndSessionButton
        onEndSession={onEndSession}
        isLoading={state === "connecting"}
      />
      <Button size="lg" isIcon variant="default" onClick={onOpen}>
        <ChevronUpIcon />
      </Button>
    </div>
  );
};

export const TogglePanel = ({
  onOpen,
  onClose,
  isOpen = false,
  state = "idle",
  onEndSession,
  collapsedButtonText,
}: TogglePanelProps) => {
  const handleOpen = () => {
    const newIsOpen = !isOpen;

    if (newIsOpen && onOpen) {
      onOpen();
    } else if (!newIsOpen && onClose) {
      onClose();
    }
  };

  return (
    <Card
      className={cn(
        "transition-all self-end z-20 duration-300 transition-discrete overflow-hidden",
        isOpen ? "shadow-xshort" : "shadow-long",
      )}
      style={{
        maxWidth: isOpen ? "100px" : "100%",
      }}
    >
      {isOpen ? (
        <Button
          size="lg"
          variant="default"
          isIcon={isOpen}
          onClick={handleOpen}
        >
          <ChevronDownIcon />
        </Button>
      ) : state !== "idle" && state !== "disconnected" ? (
        <TogglePanelSessionActive
          onEndSession={onEndSession}
          onOpen={onOpen}
          state={state}
        />
      ) : (
        <Button size="lg" variant="default" onClick={handleOpen}>
          <PhoneCallIcon />
          {collapsedButtonText}
        </Button>
      )}
    </Card>
  );
};
