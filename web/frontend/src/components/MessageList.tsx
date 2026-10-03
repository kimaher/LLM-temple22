import { ComponentPropsWithoutRef, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
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
  // Models write Markdown, so render assistant replies as such.  User text stays
  // literal: a stray `*` in a question shouldn't turn into italics.
  const markdown = role === "assistant" && !pending;

  return (
    <div className={`bubble ${role}`}>
      <div className="role">{role}</div>
      {markdown ? (
        // The caret is drawn by CSS (::after on the last block) so it stays
        // inline with the text instead of dropping below the final paragraph.
        <div className={`content markdown${streaming ? " streaming" : ""}`}>
          {/* react-markdown never renders raw HTML from the model - keep it that
              way (no rehype-raw): the output is untrusted text. */}
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex]}
            components={{ a: ExternalLink }}
          >
            {normalizeMath(content)}
          </ReactMarkdown>
        </div>
      ) : (
        <div className="content">
          {pending ? <span className="dots">generating</span> : content}
          {streaming && !pending && <span className="cursor" />}
        </div>
      )}
    </div>
  );
}

/**
 * remark-math only understands `$...$` and `$$...$$`, but models also write
 * LaTeX's `\(...\)` and `\[...\]`.  Rewrite those, and put one-line `$$ x $$`
 * on separate lines so it renders as a centred display equation rather than
 * inline.  Code is left untouched so LaTeX shown *as code* stays literal.
 */
function normalizeMath(text: string): string {
  // Odd-indexed parts are fenced code blocks or inline code spans.
  return text
    .split(/(```[\s\S]*?(?:```|$)|`[^`\n]*`)/)
    .map((part, i) =>
      i % 2 === 1
        ? part
        : part
            .replace(/\\\[([\s\S]*?)\\\]/g, (_, m: string) => `\n$$\n${m.trim()}\n$$\n`)
            .replace(/\\\(([\s\S]*?)\\\)/g, (_, m: string) => `$${m.trim()}$`)
            .replace(/^[ \t]*\$\$([^\n]+?)\$\$[ \t]*$/gm, (_, m: string) => `$$\n${m.trim()}\n$$`),
    )
    .join("");
}

/** Links in a reply open in a new tab rather than navigating away from the chat. */
function ExternalLink({ node: _node, ...props }: ComponentPropsWithoutRef<"a"> & { node?: unknown }) {
  return <a {...props} target="_blank" rel="noreferrer noopener" />;
}
