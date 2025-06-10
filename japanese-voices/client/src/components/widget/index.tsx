import { cn } from "@/lib/utils";
import { lazy, Suspense, useState } from "react";
import type { WidgetProps } from "./Widget";

const Widget = lazy(() =>
  import("./Widget").then((module) => ({ default: module.Widget })),
);

const WidgetLazy = (props: WidgetProps) => {
  const [isOpen, setIsOpen] = useState(false);
  return (
    <div className="pipecat-ui pointer-events-none" id="pipecat-ai">
      <div
        data-open={isOpen}
        className={cn(
          "widget fixed z-50 pointer-events-auto flex flex-col self-end justify-end",
          props.className,
        )}
      >
        <Suspense fallback={null}>
          <Widget {...props} onToggleOpen={setIsOpen} />
        </Suspense>
      </div>
    </div>
  );
};

export { WidgetLazy as Widget };
export type { WidgetProps };
