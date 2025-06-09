import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { VoiceVisualizer } from "@/components/widget/VoiceVisualizer";
import { MicIcon, MicOffIcon } from "@/icons";
import { RTVIEvent } from "@pipecat-ai/client-js";
import { useRTVIClient, useRTVIClientEvent } from "@pipecat-ai/client-react";
import { useCallback, useEffect, useState } from "react";

export const MicControlButton = ({
  isDisabled,
  className,
}: {
  isDisabled?: boolean;
  className?: string;
}) => {
  const client = useRTVIClient();

  const [isMicEnabled, setIsMicEnabled] = useState(
    client?.isMicEnabled ?? undefined,
  );

  useEffect(() => {
    if (!client) return;
    setIsMicEnabled(!isDisabled && client.isMicEnabled);
  }, [client, isDisabled]);

  const handleToggleMic = useCallback(() => {
    if (!client || isMicEnabled === undefined) return;

    const newEnabledState = !isMicEnabled;
    setIsMicEnabled(newEnabledState);

    if (client) {
      client.enableMic(newEnabledState);
    }
  }, [client, isMicEnabled]);

  useRTVIClientEvent(RTVIEvent.TransportStateChanged, (state) => {
    if (state === "ready") {
      setIsMicEnabled(client?.isMicEnabled);
    }
  });

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant={isMicEnabled || isDisabled ? "outline" : "destructive"}
          size="lg"
          className={className}
          onClick={handleToggleMic}
          isLoading={!client}
          disabled={isDisabled}
        >
          {client && (isMicEnabled ? <MicIcon /> : <MicOffIcon />)}
          {!isDisabled && (
            <VoiceVisualizer
              participantType="local"
              barWidth={3}
              barMaxHeight={38}
              barGap={3}
              barCount={8}
              barColor={isMicEnabled ? "#00BC7D" : "#FFFFFF"}
            />
          )}
        </Button>
      </TooltipTrigger>
      <TooltipContent>
        {isMicEnabled ? "Click to mute" : "Click to unmute"}
      </TooltipContent>
    </Tooltip>
  );
};
