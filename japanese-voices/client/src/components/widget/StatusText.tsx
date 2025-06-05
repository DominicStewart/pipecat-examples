import type { WidgetState } from "@/components/widget/Widget";
import { cn } from "@/lib/utils";
import { RTVIEvent } from "@pipecat-ai/client-js";
import { useRTVIClientEvent } from "@pipecat-ai/client-react";
import { useEffect, useState } from "react";

export const StatusText = ({ state }: { state: WidgetState }) => {
  const [statusText, setStatusText] = useState("idle");

  useEffect(() => {
    const text = {
      idle: "assistant setup",
      connecting: "connecting...",
      connected: "connected",
      disconnected: "disconnected",
    };

    setStatusText(text[state]);
  }, [state]);

  useRTVIClientEvent(RTVIEvent.BotStartedSpeaking, () => {
    setStatusText("speaking");
  });

  useRTVIClientEvent(RTVIEvent.BotStoppedSpeaking, () => {
    setStatusText("listening");
  });

  if (state === "idle") return null;

  return (
    <div
      className={cn(
        "flex mx-auto text-xs font-medium self-center justify-center bg-muted text-subtle rounded-full px-5 py-1.5 capitalize transition-all duration-500",
        statusText === "listening" && "animate-pulse",
        statusText === "speaking" && "bg-primary text-primary-foreground",
      )}
    >
      {statusText}
    </div>
  );
};
