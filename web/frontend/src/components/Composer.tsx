import { KeyboardEvent, useState } from "react";

interface Props {
  onSend: (text: string) => void;
  onStop: () => void;
  busy: boolean;
}

export function Composer({ onSend, onStop, busy }: Props) {
  const [text, setText] = useState("");

  function submit() {
    const trimmed = text.trim();
    if (!trimmed || busy) return;
    onSend(trimmed);
    setText("");
  }

  // Enter sends, Shift+Enter inserts a newline - the convention every chat UI
  // uses, so it needs no explaining to the user.
  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  return (
    <div className="composer">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder={busy ? "Generating..." : "Send a message (Enter to send, Shift+Enter for a new line)"}
        rows={2}
        disabled={busy}
      />
      {busy ? (
        <button className="stop" onClick={onStop}>
          Stop
        </button>
      ) : (
        <button className="send" onClick={submit} disabled={!text.trim()}>
          Send
        </button>
      )}
    </div>
  );
}
