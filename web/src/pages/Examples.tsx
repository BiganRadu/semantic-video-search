import CorpusPage from "../components/CorpusPage";

/** The shared corpus everyone can search: indexed footage, read-only. */
export default function Examples() {
  return (
    <CorpusPage
      collection="examples"
      title="Example videos"
      blurb="A shared corpus that is already indexed — search it to see what the engine can find."
      emptyState={
        <div className="notice">
          <b>The example corpus is empty.</b>
          <div className="muted" style={{ marginTop: 6 }}>
            Run <code>make index CORPUS=dev100</code> to populate it.
          </div>
        </div>
      }
    />
  );
}
