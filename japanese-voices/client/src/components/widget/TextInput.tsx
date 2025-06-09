import { Input } from "@/components/ui/input";
import { SendIcon } from "@/icons";
import { cn } from "@/lib/utils";
import type { LLMHelper } from "@pipecat-ai/client-js";
import { useRTVIClient } from "@pipecat-ai/client-react";
import { useState } from "react";
import { Button } from "../ui/button";

export interface TextInputProps {
  onValueChange?: (value: string) => void;
  onSend: (value: string) => void;
}

export const TextInput = ({ onValueChange, onSend }: TextInputProps) => {
  const [value, setValue] = useState("");
  const client = useRTVIClient();

  const handleValueChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setValue(e.target.value);
    onValueChange?.(e.target.value);
  };

  const handleSend = () => {
    onSend(value);
    setValue("");

    const helper = client?.getHelper("llm") as LLMHelper;
    if (!helper) return;

    helper.appendToMessages(
      {
        role: "user",
        content: value,
      },
      true,
    );
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleSend();
    }
  };

  return (
    <div className="flex-1 relative">
      <Input
        placeholder="Speak or type message here"
        size="lg"
        value={value}
        className="pr-8"
        onChange={handleValueChange}
        onKeyDown={handleKeyDown}
      />
      <Button
        variant="ghost"
        isIcon
        onClick={handleSend}
        className={cn(
          "absolute right-1 top-1 rounded-sm text-subtle/60",
          value.length > 0 && " text-primary",
        )}
      >
        <SendIcon />
      </Button>
    </div>
  );
};
