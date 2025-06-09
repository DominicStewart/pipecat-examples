import { Button } from "@/components/ui/button";
import { LoaderIcon, VolumeOffIcon } from "@/icons";

// Simple story components for component documentation

export const ButtonPrimary = ({ 
  label = "My Button",
  variant = "default",
  size = "default",
  isDisabled = false,
  isLoading = false,
  withIcon = false
}: {
  label?: string;
  variant?: "default" | "outline" | "secondary" | "ghost" | "link";
  size?: "default" | "sm" | "lg";
  isDisabled?: boolean;
  isLoading?: boolean;
  withIcon?: boolean;
}) => (
  <Button
    variant={variant}
    size={size}
    disabled={isDisabled}
    isLoading={isLoading}
  >
    {withIcon && <VolumeOffIcon />}
    {label}
  </Button>
);

export const ButtonIcon = ({ 
  variant = "default",
  size = "default",
  isDisabled = false,
  isLoading = false
}: {
  variant?: "default" | "outline" | "secondary" | "ghost" | "link";
  size?: "default" | "sm" | "lg";
  isDisabled?: boolean;
  isLoading?: boolean;
}) => (
  <Button
    isIcon
    variant={variant}
    size={size}
    disabled={isDisabled || isLoading}
  >
    {isLoading ? (
      <LoaderIcon className="size-4 animate-spin" />
    ) : (
      <VolumeOffIcon />
    )}
  </Button>
);
