import { Button } from "@/components/ui/button";
import { LoaderIcon, VolumeOffIcon } from "@/icons";
import type { Story, StoryDefault } from "@ladle/react";

export default {
  title: "Primitives",
} satisfies StoryDefault;

export const ButtonPrimary: Story<{
  label: string;
  variant: "default" | "outline" | "secondary" | "ghost" | "link";
  size: "default" | "sm" | "lg";
  isDisabled: boolean;
  isLoading: boolean;
  withIcon: boolean;
}> = ({ label, variant, size, isDisabled, isLoading, withIcon }) => (
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

ButtonPrimary.args = {
  label: "My Button",
  variant: "default",
  isDisabled: false,
  isLoading: false,
  withIcon: false,
};
ButtonPrimary.argTypes = {
  variant: {
    options: ["default", "outline", "secondary", "ghost", "link"],
    control: { type: "select" },
    defaultValue: "default",
  },
  size: {
    options: ["default", "sm", "lg"],
    control: { type: "select" },
    defaultValue: "default",
  },
};

ButtonPrimary.storyName = "Button";

export const ButtonIcon: Story<{
  variant: "default" | "outline" | "secondary" | "ghost" | "link";
  size: "default" | "sm" | "lg";
  isDisabled: boolean;
  isLoading: boolean;
}> = ({ variant, size, isDisabled, isLoading }) => (
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

ButtonIcon.args = {
  variant: "default",
  size: "default",
  isDisabled: false,
  isLoading: false,
};

ButtonIcon.argTypes = {
  variant: {
    options: ["default", "outline", "secondary", "ghost", "link"],
    control: { type: "select" },
    defaultValue: "default",
  },
  size: {
    options: ["default", "sm", "lg"],
    control: { type: "select" },
    defaultValue: "default",
  },
};

ButtonIcon.storyName = "Icon Button";
