import { SessionPanel } from "@/widget/SessionPanel";
import { TogglePanel } from "@/widget/TogglePanel";
import type {
  RTVIClientParams,
  RTVIMessage,
  TransportState,
} from "@pipecat-ai/client-js";
import { RTVIClient } from "@pipecat-ai/client-js";
import { RTVIClientAudio, RTVIClientProvider } from "@pipecat-ai/client-react";
import { DailyTransport } from "@pipecat-ai/daily-transport";
import { memo, useEffect, useRef, useState } from "react";

export interface WidgetProps {
  onConnect: () => Promise<Response>;
  collapsedButtonText?: string;
  enableTextInput?: boolean;
  className?: string;
  onToggleOpen?: (isOpen: boolean) => void;
  showTranscription?: boolean;
}

export type WidgetState = "idle" | "connecting" | "connected" | "disconnected";

const WidgetComponent = ({
  onConnect,
  collapsedButtonText = "Speak with AI assistant",
  enableTextInput = false,
  onToggleOpen,
  showTranscription = false,
}: WidgetProps) => {
  const [client, setClient] = useState<RTVIClient | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [state, setState] = useState<WidgetState>("idle");

  const isMounted = useRef(false);
  const onConnectRef = useRef(onConnect);

  onConnectRef.current = onConnect;

  useEffect(() => {
    onToggleOpen?.(isOpen);
  }, [isOpen, onToggleOpen]);

  useEffect(() => {
    if (isMounted.current) return;

    isMounted.current = true;

    const transport = new DailyTransport();

    const client = new RTVIClient({
      transport,
      enableCam: false,
      enableMic: true,
      params: {
        baseUrl: "noop",
      },
      callbacks: {
        onTransportStateChanged: (state: TransportState) => {
          switch (state) {
            case "connecting":
            case "authenticating":
            case "connected":
              setState("connecting");
              break;
            case "ready":
              setState("connected");
              break;
            case "disconnected":
            case "disconnecting":
              setState("disconnected");
              break;
            default:
              setState("idle");
              break;
          }
        },
        onError: (message: RTVIMessage) => {
          setError(message.data as string);
        },
      },
      customConnectHandler: (async (_params, timeout) => {
        try {
          const response = await onConnectRef.current();
          clearTimeout(timeout);
          if (response.ok) {
            return response.json();
          }
          const errorData = await response.text();
          setError(`Connection failed: ${response.status} ${errorData}`);
          return Promise.reject(
            new Error(`Connection failed: ${response.status}`),
          );
        } catch (err) {
          setError(
            `Connection error: ${err instanceof Error ? err.message : String(err)}`,
          );
          return Promise.reject(err);
        }
      }) as (
        params: RTVIClientParams,
        timeout: NodeJS.Timeout | undefined,
        abortController: AbortController,
      ) => Promise<void>,
    });

    setClient(client);
  }, []);

  const handleEndSession = async () => {
    await client?.disconnect();
    setIsOpen(false);
  };

  const handleStartSession = async () => {
    if (
      !client ||
      !["initialized", "disconnected", "error"].includes(client.state)
    ) {
      return;
    }
    setError(null);
    try {
      await client.connect();
    } catch (err) {
      console.error("Connection error:", err);
      setError(
        `Failed to start session: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  };

  return (
    <RTVIClientProvider client={client!}>
      {isOpen && (
        <SessionPanel
          onStartSession={handleStartSession}
          onEndSession={handleEndSession}
          error={error}
          state={state}
          enableTextInput={enableTextInput}
          showTranscription={showTranscription}
        />
      )}
      <TogglePanel
        onEndSession={handleEndSession}
        isOpen={isOpen}
        onOpen={() => setIsOpen(true)}
        onClose={() => setIsOpen(false)}
        state={state}
        collapsedButtonText={collapsedButtonText}
      />
      <RTVIClientAudio />
    </RTVIClientProvider>
  );
};

export const Widget = memo(WidgetComponent, () => true);
