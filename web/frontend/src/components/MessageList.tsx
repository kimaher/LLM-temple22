import { useEffect, useRef } from "react";
import { ChatMessage } from "../api";

interface Props {
  messages: ChatMessage[];
  /** The reply being streamed right now, or null when idle. */
  streaming: string | null;
}

export function MessageList({ messages, streaming }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Follow the stream, but only from the bottom: if the user has scrolled up to
  // read something, don't yank them back down on every token.
  useEffect(() => {
    const el = bottomRef.current?.parentElement;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    if (nearBottom) bottomRef.current?.scrollIntoView({ block: "end" });
  }, [messages, streaming]);

  const empty = messages.length === 0 && streaming === null;

  return (
    <div className="messages">
      {empty && (
        <div className="empty">
          <p>Ask the model something.</p>
          <p className="hint">
            It was trained from scratch on a small corpus, so expect small-model answers.
          </p>
        </div>
      )}

      {messages.map((m, i) => (
        <Bubble key={i} role={m.role} content={m.content} />
      ))}

      {streaming !== null && (
        <Bubble role="assistant" content={streaming} pending={streaming.length === 0} streaming />
      )}

      <div ref={bottomRef} />
    </div>
  );
}

function Bubble({
  role,
  content,
  streaming = false,
  pending = false,
}: {
  role: string;
  content: string;
  streaming?: boolean;
  pending?: boolean;
}) {
  return (
    <div className={`bubble ${role}`}>
      <div className="role">{role}</div>
      <div className="content">
        {pending ? <span className="dots">generating</span> : content}
        {streaming && !pending && <span className="cursor" />}
      </div>
    </div>
  );
}
