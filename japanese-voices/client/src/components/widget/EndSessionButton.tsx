import { Button } from "@/components/ui/button";
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
  );
};
