"""PDF Generation Skill for Genius AI.

Generates enterprise-grade, beautifully formatted PDF documents with ReportLab:
  • Cover page with title, subtitle, author, metadata, and decorative rule
  • Dynamic header & footer with 'Page X of Y' on every content page
  • Multi-level headings (H1, H2, H3) with polished typography
  • Formatted paragraphs with proper kerning and line-height
  • Styled tables with alternating zebra stripes, custom column widths, auto-wrapping
  • Highlight / Callout boxes (Info, Warning, Success)
  • Bullet points and numbered lists

Usage:
  Direct Python:
    from agent.skills.pdf_generator import generate_pdf, generate_report_pdf
    generate_pdf("report.pdf", title="Quarterly Analysis", sections=[...])

  CLI:
    python -m agent.skills.pdf_generator --output "report.pdf" --title "Executive Summary" --content "..."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    from reportlab.platypus import (
        HRFlowable,
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ─── Numbered Canvas (Page X of Y + Running Header/Footer) ────────────────────

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and stamp 'Page X of Y' and header rules."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        # Do not draw headers/footers on page 1 (cover page) if multi-page
        if self._pageNumber == 1 and page_count > 1:
            return

        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#718096"))

        # Running Header
        self.drawString(54, 11 * inch - 36, "Genius AI — Document Intelligence")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)

        # Running Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 36, page_text)
        self.drawString(54, 36, "Generated autonomously by Genius AI")
        self.line(54, 48, 8.5 * inch - 54, 48)

        self.restoreState()


# ─── PDF Document Builder ─────────────────────────────────────────────────────

def generate_pdf(
    output_path: str | Path,
    title: str,
    sections: Optional[List[Dict[str, Any]]] = None,
    subtitle: str = "",
    author: str = "Genius AI Research",
    date_str: Optional[str] = None,
    include_cover: bool = True,
    primary_color: str = "#1A365D",      # Deep Navy
    accent_color: str = "#2B6CB0",       # Slate Blue
) -> str:
    """
    Creates a professional PDF document.

    Args:
        output_path: Destination file path (e.g. 'output/report.pdf')
        title: Main document title
        sections: List of section dicts:
            [
              {"heading": "1. Overview", "text": "...", "bullets": ["a", "b"]},
              {"heading": "2. Data", "table": [["Col 1", "Col 2"], ["Val 1", "Val 2"]]},
              {"callout": "Important notice...", "callout_type": "info|warning|success"}
            ]
        subtitle: Secondary subtitle
        author: Author or organization name
        date_str: Date string (defaults to today)
        include_cover: Whether to generate a distinct cover title layout
    """
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("ReportLab is not installed. Run: pip install reportlab")

    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    date_label = date_str or datetime.now().strftime("%B %d, %Y")

    doc = SimpleDocTemplate(
        str(out_p),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=26,
        leading=32,
        textColor=colors.HexColor(primary_color),
        spaceAfter=10,
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=18,
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor(primary_color),
        spaceBefore=14,
        spaceAfter=8,
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor(accent_color),
        spaceBefore=10,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14.5,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=8,
    )

    bullet_style = ParagraphStyle(
        "BulletDark",
        parent=body_style,
        leftIndent=18,
        firstLineIndent=-10,
        spaceAfter=4,
    )

    story = []

    # 1. Header / Cover Block
    if include_cover:
        story.append(Spacer(1, 20))
        story.append(Paragraph(title, title_style))
        if subtitle:
            story.append(Paragraph(subtitle, subtitle_style))

        # Meta info
        meta_text = f"<b>Author:</b> {author} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Date:</b> {date_label}"
        story.append(Paragraph(meta_text, ParagraphStyle("Meta", parent=body_style, textColor=colors.HexColor("#718096"), fontSize=9)))
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor(accent_color), spaceAfter=18))

    # 2. Process Sections
    if sections:
        for sec in sections:
            # Heading
            if "heading" in sec:
                story.append(Paragraph(sec["heading"], h1_style))

            if "subheading" in sec:
                story.append(Paragraph(sec["subheading"], h2_style))

            # Main text
            if "text" in sec and sec["text"]:
                paragraphs = sec["text"].split("\n\n")
                for p in paragraphs:
                    if p.strip():
                        story.append(Paragraph(p.strip().replace("\n", "<br/>"), body_style))

            # Bullets
            if "bullets" in sec and isinstance(sec["bullets"], list):
                for b in sec["bullets"]:
                    story.append(Paragraph(f"• &nbsp; {b}", bullet_style))
                story.append(Spacer(1, 6))

            # Callout Box
            if "callout" in sec:
                ctype = sec.get("callout_type", "info").lower()
                c_border = "#3182CE" if ctype == "info" else ("#DD6B20" if ctype == "warning" else "#38A169")
                c_bg = "#EBF8FF" if ctype == "info" else ("#FEEBC8" if ctype == "warning" else "#F0FFF4")
                callout_p = Paragraph(f"<b>NOTE:</b> {sec['callout']}", ParagraphStyle("C", parent=body_style, textColor=colors.HexColor("#1A202C")))
                callout_table = Table([[callout_p]], colWidths=[7.0 * inch])
                callout_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(c_bg)),
                    ("LINELEFT", (0, 0), (0, -1), 3.0, colors.HexColor(c_border)),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(c_border)),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ]))
                story.append(Spacer(1, 4))
                story.append(callout_table)
                story.append(Spacer(1, 8))

            # Table
            if "table" in sec and isinstance(sec["table"], list) and len(sec["table"]) > 0:
                raw_table = sec["table"]
                # Wrap cells into Paragraphs for auto-wrap
                wrapped_table = []
                for row_idx, row in enumerate(raw_table):
                    wrapped_row = []
                    is_header = (row_idx == 0)
                    for cell in row:
                        cell_str = str(cell)
                        c_style = ParagraphStyle(
                            "TH" if is_header else "TD",
                            parent=body_style,
                            fontName="Helvetica-Bold" if is_header else "Helvetica",
                            fontSize=9.5 if is_header else 9,
                            textColor=colors.white if is_header else colors.HexColor("#2D3748"),
                            alignment=TA_CENTER if is_header else TA_LEFT,
                        )
                        wrapped_row.append(Paragraph(cell_str, c_style))
                    wrapped_table.append(wrapped_row)

                num_cols = len(raw_table[0])
                col_width = (7.0 * inch) / max(num_cols, 1)
                t = Table(wrapped_table, colWidths=[col_width] * num_cols)
                t_style = [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(primary_color)),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ]
                # Alternating row colors
                for r in range(1, len(wrapped_table)):
                    if r % 2 == 0:
                        t_style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#F7FAFC")))
                t.setStyle(TableStyle(t_style))
                story.append(Spacer(1, 6))
                story.append(t)
                story.append(Spacer(1, 10))

            if sec.get("page_break"):
                story.append(PageBreak())

    # Build document with NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    return str(out_p)


def generate_report_pdf(output_path: str, title: str, text_content: str) -> str:
    """Convenience helper to turn raw markdown/text into a styled PDF report."""
    sections = []
    current_sec = {"heading": title, "text": "", "bullets": []}

    for line in text_content.splitlines():
        line_s = line.strip()
        if line_s.startswith("# "):
            if current_sec["text"] or current_sec["bullets"]:
                sections.append(current_sec)
            current_sec = {"heading": line_s[2:].strip(), "text": "", "bullets": []}
        elif line_s.startswith("## "):
            if current_sec["text"] or current_sec["bullets"]:
                sections.append(current_sec)
            current_sec = {"heading": line_s[3:].strip(), "text": "", "bullets": []}
        elif line_s.startswith("- ") or line_s.startswith("* "):
            current_sec["bullets"].append(line_s[2:].strip())
        else:
            current_sec["text"] += line + "\n"

    if current_sec["text"] or current_sec["bullets"]:
        sections.append(current_sec)

    return generate_pdf(output_path=output_path, title=title, sections=sections)


# ─── CLI Entrypoint ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Genius AI PDF Generation Skill")
    parser.add_argument("-o", "--output", default="document.pdf", help="Output PDF path")
    parser.add_argument("-t", "--title", default="Document", help="Document Title")
    parser.add_argument("-s", "--subtitle", default="", help="Document Subtitle")
    parser.add_argument("-c", "--content", default="", help="Text or Markdown content")
    parser.add_argument("-j", "--json", default="", help="JSON string defining sections")

    args = parser.parse_args()

    if args.json:
        try:
            data = json.loads(args.json)
            sections = data if isinstance(data, list) else data.get("sections", [])
            out = generate_pdf(args.output, title=args.title, subtitle=args.subtitle, sections=sections)
            print(f"SUCCESS: Generated PDF at {out}")
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.content:
        out = generate_report_pdf(args.output, title=args.title, text_content=args.content)
        print(f"SUCCESS: Generated PDF at {out}")
    else:
        # Generate sample demo PDF
        sample_sections = [
            {
                "heading": "1. Executive Summary",
                "text": "This document was autonomously generated by Genius AI. It demonstrates professional document layout, dynamic pagination, and data presentation.",
                "callout": "All formatting, tables, and typography adhere to executive reporting standards.",
                "callout_type": "success"
            },
            {
                "heading": "2. System Performance Benchmark",
                "table": [
                    ["Metric", "Target", "Achieved", "Status"],
                    ["Reasoning Accuracy", "95.0%", "98.4%", "EXCEEDED"],
                    ["Grounding Threshold", "0.62", "0.89", "VERIFIED"],
                    ["Context Window", "32K", "1,000K", "UNLIMITED"],
                    ["Inference Latency", "<50ms", "28ms", "OPTIMAL"],
                ]
            }
        ]
        out = generate_pdf(args.output, title=args.title or "Genius AI Demo Report", subtitle="Autonomous PDF Intelligence", sections=sample_sections)
        print(f"SUCCESS: Generated sample PDF at {out}")


if __name__ == "__main__":
    main()
