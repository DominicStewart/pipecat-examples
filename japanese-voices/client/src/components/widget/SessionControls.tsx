import { useState } from "react";
import { EndSessionButton } from "./EndSessionButton";
import { MicControlButton } from "./MicControlButton";
import { TextInput } from "./TextInput";

export interface SessionControlsProps {
  onEndSession: () => void;
  isLoading?: boolean;
  isActive?: boolean;
  enabledText?: boolean;
}

export const SessionControls = ({
  onEndSession,
  isLoading,
  isActive,
  enabledText = true,
}: SessionControlsProps) => {
  const [isUsingText, setIsUsingText] = useState(false);

  if (!enabledText) {
    return (
      <footer className="flex flex-row gap-1 md:gap-2 justify-between">
        <MicControlButton isDisabled={!isActive} className="flex-1" />
        <EndSessionButton
          onEndSession={onEndSession}
          isLoading={isLoading}
          isDisabled={!isActive}
        />
      </footer>
    );
  } else {
    return (
      <footer className="flex flex-col gap-1 md:gap-2">
        <MicControlButton isDisabled={!isActive || isUsingText} />
        <div className="flex flex-row gap-1 md:gap-2 justify-between items-center">
          <TextInput
            onValueChange={(value) => setIsUsingText(value.length > 0)}
            onSend={() => {
              setIsUsingText(false);
            }}
          />
          <EndSessionButton
            onEndSession={onEndSession}
            isLoading={isLoading}
            isDisabled={!isActive}
          />
        </div>
      </footer>
    );
  }
};
