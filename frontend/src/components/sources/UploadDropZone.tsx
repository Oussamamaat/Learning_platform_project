import { useRef, useState } from "react";
import { UploadCloud } from "lucide-react";

const ACCEPT =
  ".txt,.md,.pdf,.docx,.pptx,.xlsx,.csv,.png,.jpg,.jpeg,.tiff,.tif";

interface UploadDropZoneProps {
  onFiles: (files: File[]) => void;
}

export default function UploadDropZone({ onFiles }: UploadDropZoneProps) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault(); // load-bearing: without this the browser rejects the drop entirely
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const files = Array.from(e.dataTransfer.files);
        if (files.length) onFiles(files);
      }}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
      }}
      className={`flex cursor-pointer flex-col items-center gap-1.5 rounded-xl border-2 border-dashed px-4 py-5 text-center transition-all duration-150 ease-spring ${
        dragOver
          ? "scale-[1.015] border-brand bg-brand-soft"
          : "border-edge bg-surface-2 hover:border-ink-faint/60"
      }`}
    >
      <UploadCloud className={`h-5 w-5 ${dragOver ? "text-brand" : "text-ink-faint"}`} />
      <p className="text-[12px] font-medium text-ink-dim">
        Drop files or click to upload
      </p>
      <p className="text-[10.5px] text-ink-faint">
        PDF, Word, PowerPoint, Excel, CSV, images
      </p>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={ACCEPT}
        className="hidden"
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          if (files.length) onFiles(files);
          e.target.value = ""; // reset so re-picking the same file re-fires onChange
        }}
      />
    </div>
  );
}
