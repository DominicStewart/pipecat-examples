import { cn } from "@/lib/utils";
import { RTVIEvent, type BotLLMTextData } from "@pipecat-ai/client-js";
import { useRTVIClientEvent } from "@pipecat-ai/client-react";
import { useCallback, useState } from "react";

export const TranscriptionOverlay = () => {
  const [transcript, setTranscript] = useState("");

  useRTVIClientEvent(
    RTVIEvent.BotLlmText,
    useCallback((data: BotLLMTextData) => {
      setTranscript((prev) => prev + data.text);
    }, []),
  );

  useRTVIClientEvent(
    RTVIEvent.UserStartedSpeaking,
    useCallback(() => {
      setTranscript("");
    }, []),
  );

  return (
    <div
      className={cn(
        "absolute mx-auto bottom-5 items-center justify-end text-center inset-x-0 transition-opacity duration-300",
        transcript.length > 0 && "opacity-100",
        transcript.length === 0 && "opacity-0",
      )}
    >
      <p className="box-decoration-clone leading-5 px-6 text-balance">
        <span className="bg-background/80 text-muted-foreground px-2 rounded-md text-xs box-decoration-clone text-balance">
          {transcript}
        </span>
      </p>
    </div>
  );
};
