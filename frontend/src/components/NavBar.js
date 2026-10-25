export default function NavBar() {
  return (
    <nav className="navbar" aria-label="Main navigation">
      <span className="brand">TruthLayer</span>
      <span className="nav-links">
        <a href="#verify">Verify</a>
        <span className="help" tabIndex={0} aria-label="Help">
          ?
          <span className="tooltip" role="tooltip">
            Paste AI-generated text and click Verify. Each factual claim is checked against the
            knowledge base and coloured green (Supported), red (Contradicted) or amber (Unverifiable).
          </span>
        </span>
      </span>
    </nav>
  );
}
