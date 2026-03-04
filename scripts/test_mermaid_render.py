#!/usr/bin/env python3
"""Quick test that Mermaid extraction and PNG/PDF rendering work.

Run from repo root:
  uv run python scripts/test_mermaid_render.py

Requires:
  - npm install (for @mermaid-js/mermaid-cli)
  - Node.js
  - Chrome for Puppeteer (mermaid-cli uses it to render). If you see
    "Could not find Chrome", run:
    npx puppeteer browsers install chrome-headless-shell
"""

import sys
from pathlib import Path

# Allow importing from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.visualization_agents.mermaid_render import (
    extract_mermaid_from_text,
    render_mermaid_to_png,
    render_mermaid_to_pdf,
    DIAGRAM_PNG,
    DIAGRAM_PDF,
)


def main() -> None:
    # Sample designer-like output: text with a mermaid code block
    sample_with_block = """Here is the hierarchy diagram.

```mermaid
flowchart TB
    subgraph Project["Project: Dashboard"]
        M1[Frontend Module]
        M2[Backend Module]
    end
    M1 --> T1[Task: Build UI]
    M1 --> T2[Task: Add tests]
    M2 --> T3[Task: API endpoints]
```
"""

    # 1. Extract Mermaid
    mermaid = extract_mermaid_from_text(sample_with_block)
    if not mermaid:
        print("FAIL: extract_mermaid_from_text returned None")
        sys.exit(1)
    print("OK: Extracted Mermaid source (len=%d)" % len(mermaid))
    print("---")
    print(mermaid[:200] + "..." if len(mermaid) > 200 else mermaid)
    print("---")

    # 2. Render to PNG (use a test output dir under repo)
    output_dir = Path(__file__).resolve().parent.parent / "outputs" / "mermaid_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = render_mermaid_to_png(mermaid, output_dir, DIAGRAM_PNG)
    if png_path is None or not png_path.exists():
        print("FAIL: PNG render failed.")
        print("  Ensure: npm install, then install Chrome for Puppeteer:")
        print("  npx puppeteer browsers install chrome-headless-shell")
        sys.exit(1)
    print("OK: PNG saved to", png_path)

    # 3. Render to PDF
    pdf_path = render_mermaid_to_pdf(mermaid, output_dir, DIAGRAM_PDF)
    if pdf_path is None or not pdf_path.exists():
        print("WARN: PDF render failed (optional)")
    else:
        print("OK: PDF saved to", pdf_path)

    print("\nMermaid pipeline OK. Outputs in: %s" % output_dir)


if __name__ == "__main__":
    main()
