import { useEffect, useState } from "react";
import { SearchIcon } from "./Icons";

export default function SearchBar({
  value, onSubmit, busy, placeholder,
}: {
  value: string;
  onSubmit: (q: string) => void;
  busy?: boolean;
  placeholder?: string;
}) {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);

  return (
    <form
      className="searchbar"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit(draft.trim());
      }}
    >
      <div className="field">
        <SearchIcon />
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder={placeholder ?? "a person in a red jacket carries a box out the back door"}
        />
      </div>
      <button className="primary" type="submit" disabled={busy}>
        {busy ? "searching…" : "Search"}
      </button>
    </form>
  );
}
