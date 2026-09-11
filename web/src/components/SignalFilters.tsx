import type { SignalName } from "../api/types";

const ALL: SignalName[] = ["visual", "caption", "speech", "keyword"];

/**
 * Signal overrides. A signal with nothing indexed is shown disabled rather than
 * hidden: "no data yet" and "not a feature" are different things, and a user
 * who ticks it and gets nothing would reasonably assume the search is broken.
 */
export default function SignalFilters({
  selected, coverage, onChange,
}: {
  selected: SignalName[];
  coverage?: Partial<Record<string, number>>;
  onChange: (next: SignalName[]) => void;
}) {
  const toggle = (name: SignalName) =>
    onChange(selected.includes(name) ? selected.filter((s) => s !== name) : [...selected, name]);

  return (
    <div className="filters">
      <span className="muted">signals</span>
      {ALL.map((name) => {
        const covered = coverage?.[name];
        const empty = covered === 0;
        const on = selected.includes(name);
        return (
          <label
            key={name}
            className={`toggle${on ? " on" : ""}${empty ? " disabled" : ""}`}
            title={
              covered === undefined
                ? name
                : empty
                  ? `${name}: nothing indexed in this corpus yet`
                  : `${name}: ${covered.toLocaleString()} clips indexed`
            }
          >
            <input type="checkbox" checked={on} disabled={empty} onChange={() => toggle(name)} />
            {name}
          </label>
        );
      })}
      {selected.length > 0 && (
        <button className="ghost" type="button" onClick={() => onChange([])}>
          reset
        </button>
      )}
    </div>
  );
}
