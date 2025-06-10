import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { PhoneCallEndIcon } from "@/icons";

interface EndSessionButtonProps {
  onEndSession: () => void;
  isLoading?: boolean;
  isDisabled?: boolean;
}

export const EndSessionButton = ({
  onEndSession,
  isLoading = false,
  isDisabled = false,
}: EndSessionButtonProps) => {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="outline"
          onClick={() => {
            onEndSession();
          }}
          size="lg"
          isIcon
          isLoading={isLoading}
          disabled={isDisabled}
        >
          {!isLoading && <PhoneCallEndIcon />}
        </Button>
      </TooltipTrigger>
      <TooltipContent>Disconnect</TooltipContent>
    </Tooltip>
  );
};
