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
        "absolute mx-auto items-center justify-end text-center inset-x-0 transition-opacity duration-300",
        transcript.length > 0 && "opacity-100",
        transcript.length === 0 && "opacity-0",
      )}
    >
      <p className="box-decoration-clone leading-7 px-6">
        <span className="bg-primary/40 text-primary-foreground px-2 py-1 rounded-md text-xs font-semibold box-decoration-clone">
          {transcript}
        </span>
      </p>
    </div>
  );
};
