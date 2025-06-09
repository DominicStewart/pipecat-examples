import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
} from "@/components/ui/card";
import CircularWaveform from "@/components/visualizations/CircularWaveform";
import DeviceSelect from "@/widget/DeviceSelect";
import PipecatSVG from "@/widget/PipecatSVG";
import { SessionControls } from "@/widget/SessionControls";
import { StatusText } from "@/widget/StatusText";
import type { WidgetState } from "@/widget/Widget";
import { XIcon } from "@/icons";
import { cn } from "@/lib/utils";
import { useRTVIClientMediaTrack } from "@pipecat-ai/client-react";
import { useEffect, useState } from "react";
import { TranscriptionOverlay } from "./TranscriptionOverlay";

interface SessionPanelProps {
  state: WidgetState;
  onStartSession: () => void;
  onEndSession: () => void;
  error?: string | null;
  enableTextInput?: boolean;
  showTranscription?: boolean;
}

export const SessionPanel = ({
  onEndSession,
  onStartSession,
  error,
  state,
  showTranscription = false,
}: SessionPanelProps) => {
  const agentAudioTrack = useRTVIClientMediaTrack("audio", "bot");
  const [showWaveform, setShowWaveform] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setShowWaveform(true);
    }, 300);

    return () => clearTimeout(timer);
  }, []);

  return (
    <Card className="widget-panel z-10 p-0 md:p-0 gap-0 md:gap-0 shadow-long overflow-hidden animate-widget-panel-in">
      {error && (
        <div className="bg-destructive text-white p-2 mb-4">
          An error occured, please try again later.
        </div>
      )}
      <CardHeader className="pb-0 md:pb-0">
        <StatusText state={state} />
      </CardHeader>
      <CardContent className="relative flex flex-col justify-center flex-1 w-full aspect-square md:w-widget-panel py-0 md:py-0 animate-fade-in overflow-hidden">
        {showWaveform && (
          <CircularWaveform
            audioTrack={agentAudioTrack}
            isThinking={state === "connecting"}
          />
        )}
        {showTranscription && <TranscriptionOverlay />}
      </CardContent>
      <CardContent className={cn("flex flex-col pt-0 md:pt-0")}>
        {(state === "idle" || state === "disconnected") && (
          <div className="flex flex-col gap-2 md:gap-3">
            <DeviceSelect />
            <Button onClick={onStartSession} size="lg">
              {state === "idle" ? "Let's talk!" : "Talk again"}
            </Button>
          </div>
        )}
        {state === "connected" && (
          <SessionControls
            onEndSession={onEndSession}
            isActive={state === "connected"}
          />
        )}
      </CardContent>
      <CardFooter className="bg-secondary border-t py-1.5 md:py-2">
        <div className="flex flex-row gap-1.5 items-center justify-between w-full">
          <span className="text-[10px] uppercase tracking-wider text-subtle font-medium">
            Built with:
          </span>
          <div className="flex flex-row gap-1.5 items-center">
            <PipecatSVG className="h-[15px] w-auto text-card-foreground" />
            <XIcon className="size-3 text-subtle/60" />
            <span className="text-sm font-medium text-card-foreground">Gemini</span>
          </div>
        </div>
      </CardFooter>
    </Card>
  );
};
