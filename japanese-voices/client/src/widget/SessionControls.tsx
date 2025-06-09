import { EndSessionButton } from "./EndSessionButton";
import { MicControlButton } from "./MicControlButton";

export interface SessionControlsProps {
  onEndSession: () => void;
  isLoading?: boolean;
  isActive?: boolean;
}

export const SessionControls = ({
  onEndSession,
  isLoading,
  isActive,
}: SessionControlsProps) => {
  return (
    <footer className="flex flex-row gap-1 md:gap-2 justify-between">
      <MicControlButton isDisabled={!isActive} />
      <EndSessionButton
        onEndSession={onEndSession}
        isLoading={isLoading}
        isDisabled={!isActive}
      />
    </footer>
  );
};
