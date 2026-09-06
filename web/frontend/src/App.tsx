import { useEffect, useRef, useState } from "react";
import { ChatMessage, HealthInfo, getHealth, streamChat } from "./api";
import { MessageList } from "./components/MessageList";
import { Composer } from "./components/Composer";
import { SettingsBar, Settings, DEFAULT_SETTINGS } from "./components/SettingsBar";

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  // The reply currently being streamed, kept out of `messages` so a re-render
  // per token doesn't touch the finished history.
  const [streaming, setStreaming] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const abortRef = useRef<AbortController | null>(null);
  const busy = streaming !== null;

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  async function send(text: string) {
    const history: ChatMessage[] = [...messages, { role: "user", content: text }];
    setMessages(history);
    setStreaming("");
    setError(null);

    const controller = new AbortController();
    abortRef.current = controller;
    let reply = "";

    try {
      for await (const event of streamChat(history, settings, controller.signal)) {
        if (event.type === "token") {
          reply += event.text;
          setStreaming(reply);
        } else if (event.type === "error") {
          setError(event.message);
        }
      }
    } catch (err) {
      // An abort is a deliberate stop, not a failure: keep what we streamed.
      if ((err as Error).name !== "AbortError") setError(String(err));
    } finally {
      abortRef.current = null;
      setStreaming(null);
      if (reply.trim().length > 0) {
        setMessages([...history, { role: "assistant", content: reply }]);
      }
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  function reset() {
    stop();
    setMessages([]);
    setError(null);
  }

  const engineLabel = health
    ? health.engine === "mock"
      ? "mock engine - no checkpoint loaded"
      : `${((health.info.params as number) / 1e6).toFixed(1)}M params on ${health.info.device}`
    : "backend offline";

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>LLM-temple22</h1>
          <span className={`badge ${health?.engine === "model" ? "badge-live" : "badge-mock"}`}>
            {engineLabel}
          </span>
        </div>
        <button className="ghost" onClick={reset} disabled={messages.length === 0 && !busy}>
          New chat
        </button>
      </header>

      <SettingsBar settings={settings} onChange={setSettings} disabled={busy} />

      <MessageList messages={messages} streaming={streaming} />

      {error && <div className="error">{error}</div>}

      <Composer onSend={send} onStop={stop} busy={busy} />
    </div>
  );
}
