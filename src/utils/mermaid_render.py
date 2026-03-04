"""Render Mermaid diagram source to PNG (and optionally PDF)."""

import logging
import re
import subprocess
import tempfile
from pathlib import Path

console_logger = logging.getLogger(__name__)

# Default output filenames
DIAGRAM_PNG = "hierarchy_diagram.png"
DIAGRAM_PDF = "hierarchy_diagram.pdf"
DIAGRAM_MMD = "hierarchy_diagram.mmd"

# Default quality: scale 3 = 3x resolution; large viewport for big diagrams
DEFAULT_SCALE = 3
DEFAULT_WIDTH = 2400
DEFAULT_HEIGHT = 1800


def _quote_bracket_content(content: str) -> str:
    """Escape and wrap for Mermaid: ["..."] with internal " and \ escaped."""
    escaped = content.replace("\\", "\\\\").replace('"', '\\"')
    return f'["{escaped}"]'


def sanitize_mermaid_for_render(mermaid_source: str) -> str:
    """Apply minimal fixes so mmdc can parse LLM-generated Mermaid (e.g. quote labels with parens/slashes)."""
    if not mermaid_source or not mermaid_source.strip():
        return mermaid_source

    def quote_if_needed(match: re.Match) -> str:
        content = match.group(1)
        needs = (
            "(" in content or ")" in content or "/" in content or "|" in content or "`" in content or " " in content
        )
        return _quote_bracket_content(content) if needs else match.group(0)

    s = re.sub(r"\[([^\]\"]+)\]", quote_if_needed, mermaid_source)

    # Split lines that have multiple arrows (e.g. "G -->|Uses D|         G -->|Compl").
    # Parser expects one statement per line; put each arrow on its own line.
    lines = s.split("\n")
    out = []
    for line in lines:
        # If line contains "-->", split at "whitespace + node_id + -->" so second arrow starts on new line
        if "-->" in line and line.strip().startswith(("flowchart", "graph", "subgraph", "end")) is False:
            # After first "-->|label|" or "-->", if we see more spaces and then ID -->, break to newline
            fixed = re.sub(
                r"(-->\|[^|]*\|)\s+([A-Za-z0-9_\"\[]+)\s*-->",
                r"\1\n\2 -->",
                line,
            )
            # Same for arrow without label: "-->  NODE -->" -> "--> \nNODE -->"
            fixed = re.sub(
                r"(-->)\s{2,}([A-Za-z0-9_\"\[]+)\s*-->",
                r"\1\n\2 -->",
                fixed,
            )
            out.append(fixed)
        else:
            out.append(line)

    return "\n".join(out)


def extract_mermaid_from_text(text: str) -> str | None:
    """Extract Mermaid code block from designer output.

    Looks for ```mermaid ... ``` or ``` ... ``` and returns the inner content.
    """
    if not text or not text.strip():
        return None
    # Prefer explicit mermaid block
    mermaid_match = re.search(
        r"```mermaid\s*\n(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if mermaid_match:
        return mermaid_match.group(1).strip()
    # Fallback: generic code block (designer might output "here is the diagram")
    code_match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    # No block: treat whole text as Mermaid if it looks like it (flowchart, graph, etc.)
    stripped = text.strip()
    if stripped.startswith("flowchart") or stripped.startswith("graph "):
        return stripped
    return None


def render_mermaid_to_png(
    mermaid_source: str,
    output_dir: Path,
    output_name: str = DIAGRAM_PNG,
    scale: int = DEFAULT_SCALE,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> Path | None:
    """Render Mermaid source to PNG using mermaid-cli (mmdc).

    Requires Node.js and npx, or a globally installed @mermaid-js/mermaid-cli.

    Args:
        mermaid_source: Raw Mermaid diagram source.
        output_dir: Directory to write the PNG and temporary .mmd file.
        output_name: Output filename (default hierarchy_diagram.png).
        scale: Puppeteer scale factor (default 2 for higher DPI).
        width: Page width in pixels.
        height: Page height in pixels.

    Returns:
        Path to the generated PNG, or None if rendering failed (e.g. mmdc not found).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mmd_path = output_dir / DIAGRAM_MMD
    png_path = output_dir / output_name

    sanitized = sanitize_mermaid_for_render(mermaid_source)
    mmd_path.write_text(sanitized, encoding="utf-8")

    try:
        cmd = [
            "npx",
            "-y",
            "@mermaid-js/mermaid-cli",
            "-i",
            str(mmd_path),
            "-o",
            str(png_path),
            "-s",
            str(scale),
            "-w",
            str(width),
            "-H",
            str(height),
        ]
        # Don't capture stdout/stderr so mmdc output appears in the same terminal as the main process
        proc = subprocess.run(
            cmd,
            timeout=60,
        )
        if proc.returncode != 0:
            console_logger.warning("Mermaid PNG render failed (exit %s)", proc.returncode)
            return None
        console_logger.info(f"Rendered Mermaid to {png_path}")
        return png_path
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        console_logger.warning("Mermaid PNG render failed: %s", e)
        return None


def render_mermaid_to_pdf(
    mermaid_source: str,
    output_dir: Path,
    output_name: str = DIAGRAM_PDF,
    scale: int = DEFAULT_SCALE,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    pdf_fit: bool = True,
) -> Path | None:
    """Render Mermaid source to PDF using mermaid-cli (mmdc).

    Args:
        mermaid_source: Raw Mermaid diagram source.
        output_dir: Directory to write the PDF and temporary .mmd file.
        output_name: Output filename (default hierarchy_diagram.pdf).
        scale: Puppeteer scale factor (default 2 for higher DPI).
        width: Page width in pixels.
        height: Page height in pixels.
        pdf_fit: If True, pass -f so PDF is scaled to fit the chart.

    Returns:
        Path to the generated PDF, or None if rendering failed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mmd_path = output_dir / DIAGRAM_MMD
    pdf_path = output_dir / output_name

    sanitized = sanitize_mermaid_for_render(mermaid_source)
    mmd_path.write_text(sanitized, encoding="utf-8")

    try:
        cmd = [
            "npx",
            "-y",
            "@mermaid-js/mermaid-cli",
            "-i",
            str(mmd_path),
            "-o",
            str(pdf_path),
            "-s",
            str(scale),
            "-w",
            str(width),
            "-H",
            str(height),
        ]
        if pdf_fit:
            cmd.append("-f")
        # Don't capture stdout/stderr so mmdc output appears in the same terminal as the main process
        proc = subprocess.run(
            cmd,
            timeout=60,
        )
        if proc.returncode != 0:
            console_logger.warning("Mermaid PDF render failed (exit %s)", proc.returncode)
            return None
        console_logger.info(f"Rendered Mermaid to {pdf_path}")
        return pdf_path
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        console_logger.warning("Mermaid PDF render failed: %s", e)
        return None
