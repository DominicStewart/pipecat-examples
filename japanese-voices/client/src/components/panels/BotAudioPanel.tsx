import { Button } from "@/components/ui/button";
import {
  Panel,
  PanelActions,
  PanelContent,
  PanelFooter,
  PanelHeader,
  PanelTitle,
} from "@/components/ui/panel";
import {
  Select,
  SelectContent,
  SelectGuide,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { EllipsisIcon, VolumeIcon, VolumeOffIcon } from "@/icons";

interface BotAudioPanelProps {
  audioTracks?: MediaStreamTrack[];
  visualization?: "bar" | "circle";
  isMuted?: boolean;
  onMuteToggle?: () => void;
}

export const BotAudioPanel: React.FC<BotAudioPanelProps> = ({
  isMuted = false,
  onMuteToggle,
}) => {
  return (
    <Panel>
      <PanelHeader>
        <PanelTitle>Bot Audio</PanelTitle>
      </PanelHeader>
      <PanelContent>
        <div className="flex items-center gap-2 justify-between">
          <div className="flex flex-row items-center gap-1 text-xs font-mono">
            <span className="text-muted-foreground @max-sm/panel:hidden">
              Status:
            </span>
            <span className="text-foreground">Listening</span>
          </div>
          <PanelActions>
            <Button
              isIcon
              variant={isMuted ? "muted" : "ghost"}
              onClick={onMuteToggle}
            >
              {isMuted ? <VolumeOffIcon /> : <VolumeIcon />}
            </Button>
            <Button isIcon variant="outline">
              <EllipsisIcon />
            </Button>
          </PanelActions>
        </div>
        <div className="aspect-video bg-muted rounded-sm"></div>
      </PanelContent>
      <PanelFooter className="border-t">
        <Select defaultValue="track-1">
          <SelectTrigger size="sm">
            <SelectGuide>Track:</SelectGuide>
            <SelectValue placeholder="Select a track" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="track-1">Track 1</SelectItem>
            <SelectItem value="track-2">Track 2</SelectItem>
            <SelectItem value="track-3">Track 3</SelectItem>
          </SelectContent>
        </Select>
      </PanelFooter>
    </Panel>
  );
};
