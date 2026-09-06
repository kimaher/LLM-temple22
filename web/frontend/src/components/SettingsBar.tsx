import { GenerationOptions } from "../api";

export type Settings = Required<Omit<GenerationOptions, "seed">>;

export const DEFAULT_SETTINGS: Settings = {
  max_new_tokens: 256,
  temperature: 0.8,
  top_k: 50,
  top_p: 0.95,
  repetition_penalty: 1.1,
};

interface Props {
  settings: Settings;
  onChange: (s: Settings) => void;
  disabled: boolean;
}

/** Exposes the sampling knobs, which is half the fun of running your own model. */
export function SettingsBar({ settings, onChange, disabled }: Props) {
  function set<K extends keyof Settings>(key: K, value: number) {
    onChange({ ...settings, [key]: value });
  }

  return (
    <div className="settings">
      <Slider
        label="temperature"
        value={settings.temperature}
        min={0}
        max={1.5}
        step={0.05}
        disabled={disabled}
        onChange={(v) => set("temperature", v)}
        hint="0 = greedy"
      />
      <Slider
        label="top-k"
        value={settings.top_k}
        min={0}
        max={200}
        step={1}
        disabled={disabled}
        onChange={(v) => set("top_k", v)}
        hint="0 = off"
      />
      <Slider
        label="top-p"
        value={settings.top_p}
        min={0.1}
        max={1}
        step={0.01}
        disabled={disabled}
        onChange={(v) => set("top_p", v)}
      />
      <Slider
        label="max tokens"
        value={settings.max_new_tokens}
        min={16}
        max={1024}
        step={16}
        disabled={disabled}
        onChange={(v) => set("max_new_tokens", v)}
      />
    </div>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  disabled,
  hint,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  disabled: boolean;
  hint?: string;
}) {
  return (
    <label className="slider" title={hint}>
      <span>
        {label} <b>{value}</b>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}
