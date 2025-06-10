import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";

export const ConsoleLayout = () => {
  return (
    <div className="grid grid-cols-1 grid-rows-[1fr_auto] h-screen w-full gap-4 bg-red-200">
      <ResizablePanelGroup direction="horizontal" className="">
        <ResizablePanel defaultSize={30} minSize={30}>
          <div className="h-full bg-green-500 p-4">Tracks</div>
        </ResizablePanel>
        <ResizableHandle withHandle />
        <ResizablePanel defaultSize={50} minSize={40}>
          <div className="h-full bg-indigo-500 p-4">Conversation</div>
        </ResizablePanel>
        <ResizableHandle withHandle />
        <ResizablePanel
          collapsible={true}
          collapsedSize={5}
          minSize={20}
          onCollapse={() => {
            console.log("collapsed");
          }}
        >
          <div className="h-full bg-yellow-500 p-4">Aside</div>
        </ResizablePanel>
      </ResizablePanelGroup>
      <div className="bg-pink-500 p-4 text-gray-500 min-h-[100px]">console</div>
    </div>
  );
};
