"""HTML report: every experiment and score in PostgreSQL -> one self-contained page, no server needed."""
import json
from datetime import datetime
from importlib.resources import files

from hospital_rag import db
from hospital_rag.config import settings


def load_report_data(conn):
    """Experiments, raw scores and eval questions. The page computes the means itself, so it can regroup freely."""
    experiments = [
        {
            "id": id_, "name": name, "config": config, "status": status, "duration_s": duration_s,
            "started_at": started_at.isoformat(timespec="minutes"),
        }
        for id_, name, config, started_at, duration_s, status in db.list_experiments(conn)
    ]
    with open(settings.eval_path, encoding="utf-8") as f:
        questions = {
            q["id"]: {"type": q["type"], "question": q["question"], "answer": q["answer"]} for q in json.load(f)
        }
    return {
        "generated_at": datetime.now().isoformat(timespec="minutes"),
        "experiments": experiments,
        "scores": [list(row) for row in db.list_scores(conn)],  # lists, not dicts: a third of the size
        "questions": questions,
    }


def render_report(data):
    """The template with the data inlined as JSON."""
    template = files("hospital_rag").joinpath("templates", "report.html").read_text(encoding="utf-8")
    # "</" inside a <script> would end it early if an answer or a prompt ever contains "</script>"
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return template.replace("/*__REPORT_DATA__*/null", payload)


def write_report(path=None):
    """Read the database and write the report. Returns the path of the HTML file."""
    path = path or settings.reports_path / "metrics.html"
    with db.connect() as conn:
        data = load_report_data(conn)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(data), encoding="utf-8")
    return path
