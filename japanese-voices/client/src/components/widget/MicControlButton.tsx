import { Button } from "@/components/ui/button";
import { VoiceVisualizer } from "@/components/widget/VoiceVisualizer";
import { MicIcon, MicOffIcon } from "@/icons";
import { RTVIEvent } from "@pipecat-ai/client-js";
import { useRTVIClient, useRTVIClientEvent } from "@pipecat-ai/client-react";
import { useCallback, useEffect, useState } from "react";

export const MicControlButton = ({ isDisabled }: { isDisabled?: boolean }) => {
  const client = useRTVIClient();

  const [isMicEnabled, setIsMicEnabled] = useState(
    client?.isMicEnabled ?? undefined,
  );

  useEffect(() => {
    if (!client) return;
    setIsMicEnabled(client.isMicEnabled);
  }, [client]);

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
    <Button
      variant={isMicEnabled || isDisabled ? "outline" : "destructive"}
      size="lg"
      className="flex-1"
      onClick={handleToggleMic}
      isLoading={!client}
      disabled={isDisabled}
    >
      {client && (isMicEnabled ? <MicIcon /> : <MicOffIcon />)}
      <VoiceVisualizer
        participantType="local"
        barWidth={3}
        barMaxHeight={38}
        barGap={3}
        barCount={8}
        barColor={isMicEnabled ? "#00BC7D" : "#FFFFFF"}
      />
    </Button>
  );
};
