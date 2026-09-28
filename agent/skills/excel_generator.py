"""Excel Spreadsheet Generation Skill for Genius AI.

Generates enterprise-grade, beautifully formatted .xlsx workbooks using openpyxl:
  • Multi-sheet workbooks with custom tab colors
  • Themed header rows (Deep Navy, Emerald, Charcoal, Slate) with bold white text
  • Auto-calculated column widths with safety margins (zero truncation or '###' errors)
  • Explicit number, currency, percentage, and date cell formatting
  • Formula injection (SUM, AVERAGE, MIN, MAX, COUNT, ratios)
  • Summary / Total rows with classic double-underline accounting borders
  • Alternating zebra row shading for high readability
  • Grid lines explicitly preserved

Usage:
  Direct Python:
    from agent.skills.excel_generator import generate_excel
    generate_excel("sales.xlsx", sheets=[...])

  CLI:
    python -m agent.skills.excel_generator --output "sales.xlsx" --title "Quarterly Sales"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


# ─── Color Themes ─────────────────────────────────────────────────────────────

THEMES = {
    "navy": {
        "header_fill": "1B365D",
        "header_font": "FFFFFF",
        "zebra_fill": "F0F4F8",
        "accent_fill": "D9E2EC",
        "total_fill": "E2E8F0",
    },
    "emerald": {
        "header_fill": "1C4E35",
        "header_font": "FFFFFF",
        "zebra_fill": "F0FFF4",
        "accent_fill": "C6F6D5",
        "total_fill": "E6FFFA",
    },
    "charcoal": {
        "header_fill": "2D3748",
        "header_font": "FFFFFF",
        "zebra_fill": "F7FAFC",
        "accent_fill": "E2E8F0",
        "total_fill": "EDF2F7",
    },
    "wine": {
        "header_fill": "5B1E31",
        "header_font": "FFFFFF",
        "zebra_fill": "FFF5F5",
        "accent_fill": "FED7D7",
        "total_fill": "FEEBC8",
    },
}


def _get_borders() -> Tuple[Border, Border, Border]:
    thin_side = Side(style="thin", color="CBD5E0")
    double_side = Side(style="double", color="1A202C")
    thin_top = Side(style="thin", color="1A202C")

    cell_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)
    header_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=Side(style="medium", color="1A202C"))
    total_border = Border(top=thin_top, bottom=double_side)

    return cell_border, header_border, total_border


# ─── Excel Document Builder ───────────────────────────────────────────────────

def generate_excel(
    output_path: str | Path,
    sheets: List[Dict[str, Any]],
    theme: str = "navy",
    freeze_panes: bool = True,
) -> str:
    """
    Creates an Excel .xlsx workbook.

    Args:
        output_path: Destination path (e.g. 'sales_report.xlsx')
        sheets: List of sheet definitions:
            [
              {
                "title": "Q1 Performance",
                "headers": ["Region", "Sales", "Expenses", "Net Profit", "Margin %"],
                "rows": [
                    ["North", 150000, 90000, "=B2-C2", "=D2/B2"],
                    ["South", 220000, 130000, "=B3-C3", "=D3/B3"],
                ],
                "formats": {"B": "currency", "C": "currency", "D": "currency", "E": "percent"},
                "totals": ["Total", "=SUM(B2:B3)", "=SUM(C2:C3)", "=SUM(D2:D3)", "=D4/B4"],
              }
            ]
        theme: Color theme ('navy', 'emerald', 'charcoal', 'wine')
        freeze_panes: Freeze top header row so it stays visible during scroll
    """
    if not OPENPYXL_AVAILABLE:
        raise RuntimeError("openpyxl is not installed. Run: pip install openpyxl")

    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    theme_cfg = THEMES.get(theme.lower(), THEMES["navy"])
    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    cell_border, header_border, total_border = _get_borders()

    for sheet_idx, s_def in enumerate(sheets):
        sheet_title = s_def.get("title", f"Sheet{sheet_idx+1}")[:31]  # Excel max 31 chars
        ws = wb.create_sheet(title=sheet_title)

        # Force grid lines visible
        ws.views.sheetView[0].showGridLines = True

        headers = s_def.get("headers", [])
        rows = s_def.get("rows", [])
        formats = s_def.get("formats", {})
        totals = s_def.get("totals", None)

        current_row = 1

        # Optional Title Banner inside sheet
        banner_title = s_def.get("banner", "")
        if banner_title:
            ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=len(headers) or 4)
            top_cell = ws.cell(row=current_row, column=1, value=banner_title)
            top_cell.font = Font(name="Segoe UI", size=14, bold=True, color="1A365D")
            top_cell.alignment = Alignment(vertical="center", horizontal="left")
            ws.row_dimensions[current_row].height = 28
            current_row += 2  # Leave space before table

        header_row_num = current_row

        # 1. Write Headers
        if headers:
            ws.row_dimensions[header_row_num].height = 24
            for col_idx, h_text in enumerate(headers, start=1):
                cell = ws.cell(row=header_row_num, column=col_idx, value=str(h_text))
                cell.font = Font(name="Segoe UI", size=10.5, bold=True, color=theme_cfg["header_font"])
                cell.fill = PatternFill(start_color=theme_cfg["header_fill"], end_color=theme_cfg["header_fill"], fill_type="solid")
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                cell.border = header_border

            current_row += 1

        # 2. Write Data Rows
        zebra_fill = PatternFill(start_color=theme_cfg["zebra_fill"], end_color=theme_cfg["zebra_fill"], fill_type="solid")

        data_start_row = current_row
        for row_data in rows:
            ws.row_dimensions[current_row].height = 20
            is_even = (current_row % 2 == 0)

            for col_idx, val in enumerate(row_data, start=1):
                cell = ws.cell(row=current_row, column=col_idx)
                col_letter = get_column_letter(col_idx)
                fmt_type = formats.get(col_letter, formats.get(str(col_idx), ""))

                # Assign value
                if isinstance(val, str) and val.startswith("="):
                    cell.value = val  # Formula
                else:
                    cell.value = val

                cell.font = Font(name="Segoe UI", size=10, color="2D3748")
                cell.border = cell_border
                if is_even:
                    cell.fill = zebra_fill

                # Alignments & Formatting
                if isinstance(val, (int, float)):
                    cell.alignment = Alignment(horizontal="right", vertical="center")
                    if fmt_type == "currency":
                        cell.number_format = "₹#,##0.00"
                    elif fmt_type == "percent":
                        cell.number_format = "0.0%"
                    elif fmt_type == "integer":
                        cell.number_format = "#,##0"
                    else:
                        cell.number_format = "#,##0.00" if isinstance(val, float) else "#,##0"
                elif isinstance(val, str) and (val.startswith("=") or fmt_type == "percent"):
                    cell.alignment = Alignment(horizontal="right", vertical="center")
                    if fmt_type == "percent":
                        cell.number_format = "0.0%"
                    elif fmt_type == "currency":
                        cell.number_format = "₹#,##0.00"
                else:
                    cell.alignment = Alignment(horizontal="left", vertical="center")

            current_row += 1

        # 3. Write Totals / Summary Row
        if totals:
            ws.row_dimensions[current_row].height = 22
            total_bg = PatternFill(start_color=theme_cfg["total_fill"], end_color=theme_cfg["total_fill"], fill_type="solid")

            for col_idx, t_val in enumerate(totals, start=1):
                cell = ws.cell(row=current_row, column=col_idx, value=t_val)
                cell.font = Font(name="Segoe UI", size=10.5, bold=True, color="1A202C")
                cell.fill = total_bg
                cell.border = total_border

                col_letter = get_column_letter(col_idx)
                fmt_type = formats.get(col_letter, formats.get(str(col_idx), ""))
                if fmt_type == "currency":
                    cell.number_format = "₹#,##0.00"
                    cell.alignment = Alignment(horizontal="right", vertical="center")
                elif fmt_type == "percent":
                    cell.number_format = "0.0%"
                    cell.alignment = Alignment(horizontal="right", vertical="center")
                else:
                    cell.alignment = Alignment(horizontal="right" if isinstance(t_val, (int, float)) or (isinstance(t_val, str) and t_val.startswith("=")) else "left", vertical="center")

            current_row += 1

        # 4. Auto-Fit Column Widths
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                val = cell.value
                if val:
                    # Ignore merged title row for width calculations
                    if cell.row == 1 and banner_title:
                        continue
                    v_str = str(val)
                    if v_str.startswith("="):
                        v_str = "123,456.78"  # estimate formula width
                    max_len = max(max_len, len(v_str))
            ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

        # 5. Freeze Panes
        if freeze_panes and headers:
            ws.freeze_panes = ws.cell(row=header_row_num + 1, column=1)

    wb.save(str(out_p))
    return str(out_p)


# ─── CLI Entrypoint ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Genius AI Excel Generation Skill")
    parser.add_argument("-o", "--output", default="spreadsheet.xlsx", help="Output .xlsx file path")
    parser.add_argument("-t", "--title", default="Report", help="Sheet Title")
    parser.add_argument("-m", "--theme", default="navy", choices=["navy", "emerald", "charcoal", "wine"], help="Color theme")
    parser.add_argument("-j", "--json", default="", help="JSON string defining sheets data")

    args = parser.parse_args()

    if args.json:
        try:
            data = json.loads(args.json)
            sheets = data if isinstance(data, list) else [data]
            out = generate_excel(args.output, sheets=sheets, theme=args.theme)
            print(f"SUCCESS: Generated Excel spreadsheet at {out}")
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Default sample financial spreadsheet
        sample_sheets = [
            {
                "title": "FY26 Revenue",
                "banner": "Genius AI — Enterprise Financial Projection",
                "headers": ["Department", "Q1 Actual", "Q2 Actual", "Q3 Projected", "Q4 Projected", "Annual Total"],
                "rows": [
                    ["Cloud Infrastructure", 450000, 520000, 610000, 700000, "=SUM(B4:E4)"],
                    ["AI Research & Core", 280000, 340000, 420000, 510000, "=SUM(B5:E5)"],
                    ["Security & Compliance", 120000, 135000, 150000, 165000, "=SUM(B6:E6)"],
                    ["Developer Operations", 95000, 110000, 125000, 140000, "=SUM(B7:E7)"],
                ],
                "formats": {"B": "currency", "C": "currency", "D": "currency", "E": "currency", "F": "currency"},
                "totals": ["Total Revenue", "=SUM(B4:B7)", "=SUM(C4:C7)", "=SUM(D4:D7)", "=SUM(E4:E7)", "=SUM(F4:F7)"],
            }
        ]
        out = generate_excel(args.output, sheets=sample_sheets, theme=args.theme)
        print(f"SUCCESS: Generated sample Excel spreadsheet at {out}")


if __name__ == "__main__":
    main()
