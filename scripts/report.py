"""Write the HTML report of every experiment in PostgreSQL. Open the file in a browser.

Usage: uv run python scripts/report.py [output.html]
"""
import sys
from pathlib import Path

from hospital_rag.evaluation.report import write_report

path = write_report(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
print(f"file://{path.resolve()}")
