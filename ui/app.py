"""Results dashboard: leaderboard, comparison of two experiments, detail of one question. Reads PostgreSQL only.

Run from the repository root: uv run streamlit run ui/app.py
"""
import json

import pandas as pd
import streamlit as st

from hospital_rag import db
from hospital_rag.config import settings
from hospital_rag.evaluation.metrics import judge_chunk
from hospital_rag.evaluation.report import load_report_data
from hospital_rag.evaluation.runner import flatten

st.set_page_config(page_title="RAG experiments", layout="wide")

FAMILIES = ["hit", "mrr", "recall", "ndcg"]  # retrieval metrics first, in this order; the others after, alphabetically


def metric_order(metric):
    family = next((i for i, f in enumerate(FAMILIES) if metric.startswith(f)), len(FAMILIES))
    k = int(metric.split("@")[1]) if "@" in metric else 0
    return family, k, metric


@st.cache_data(ttl=300)
def load():
    """Experiments, scores (one row per question and metric) and questions, from a read-only connection."""
    with db.connect(read_only=True) as conn:
        data = load_report_data(conn)
    experiments = pd.DataFrame(data["experiments"])
    experiments = experiments[experiments["status"] == "done"]
    scores = pd.DataFrame(data["scores"], columns=["experiment_id", "question_id", "question_type", "metric", "value"])
    scores = scores[scores["experiment_id"].isin(experiments["id"])].dropna(subset=["value"])
    # Refusals mean "good" on out_of_corpus questions and "wrong" on the others: split them into two metrics
    refusal = scores[scores["metric"] == "refusal"]
    split = refusal.assign(metric=refusal["question_type"].map(
        lambda t: "out_of_corpus refused" if t == "out_of_corpus" else "wrong refusals"))
    scores = pd.concat([scores[scores["metric"] != "refusal"], split])
    scores = scores.merge(experiments[["id", "name"]], left_on="experiment_id", right_on="id").drop(columns="id")
    return experiments, scores, data["questions"]


def per_question(scores, name, metric):
    """Mean over the runs of an experiment, question by question."""
    rows = scores[(scores["name"] == name) & (scores["metric"] == metric)]
    return rows.groupby(["question_id", "question_type"])["value"].mean()


def config_changes(experiments, name, reference):
    """The config keys that differ from the reference experiment (its most recent run), as "key = value"."""
    last = lambda n: flatten(experiments[experiments["name"] == n].iloc[-1]["config"])
    if name == reference or reference not in set(experiments["name"]):
        return ""
    a, b = last(reference), last(name)
    changed = sorted(k for k in a.keys() | b.keys() if k != "name" and a.get(k) != b.get(k))
    show = lambda v: "(changed)" if isinstance(v, str) and len(v) > 40 else repr(v)
    return ", ".join(f"{k} = {show(b.get(k))}" for k in changed)


@st.cache_data
def load_golden():
    """The golden dataset with its evidence (load_report_data keeps only the question and the expected answer)."""
    with open(settings.eval_path, encoding="utf-8") as f:
        return {q["id"]: q for q in json.load(f)}


experiments, scores, questions = load()
golden = load_golden()
names = list(dict.fromkeys(experiments["name"]))  # in order of first run
metrics = sorted(scores["metric"].unique(), key=metric_order)
reference = "baseline" if "baseline" in names else names[0]

with st.sidebar:
    st.title("RAG experiments")
    page = st.radio("Page", ["Leaderboard", "Compare two experiments", "Question detail"])
    st.caption(f"{len(experiments)} finished runs · {len(names)} experiments · reference: **{reference}**")
    if st.button("Reload from the database"):
        load.clear()
        st.rerun()

if page == "Leaderboard":
    st.header("Leaderboard")
    default = [m for m in ["recall@5", "mrr@5", "ndcg@5", "out_of_corpus refused", "wrong refusals",
                           "faithfulness", "correctness"] if m in metrics]
    chosen = st.multiselect("Metrics", metrics, default=default)
    sort_by = st.selectbox("Sort by", chosen or metrics)
    # Mean over the questions for each run, then over the runs of an experiment (runs with the same name are repeats)
    run_means = scores[scores["metric"].isin(chosen)].groupby(["name", "experiment_id", "metric"])["value"].mean()
    table = run_means.groupby(["name", "metric"]).mean().unstack("metric").reindex(index=names, columns=chosen)
    table.insert(0, "runs", experiments.groupby("name")["id"].count())
    table["changes vs " + reference] = [config_changes(experiments, n, reference) for n in table.index]
    table = table.sort_values(sort_by, ascending=sort_by == "wrong refusals", na_position="last")
    st.caption("Mean over the questions, then over the runs. Refusals are rates (wrong refusals: share of the 100 "
               "answerable questions refused). Highlighted: best of each column (lowest for wrong refusals). "
               "Judge metrics: 26-question sample; retrieval: 100 questions; – : not measured.")
    higher = [m for m in chosen if m != "wrong refusals"]
    styled = (table.style.format({m: "{:.3f}" for m in chosen}, na_rep="–")
              .highlight_max(subset=higher, color="#2f6fdf55").highlight_min(
                  subset=[m for m in chosen if m == "wrong refusals"], color="#2f6fdf55"))
    st.dataframe(styled, width="stretch", height=min(38 * (len(table) + 1), 900))
    st.bar_chart(table[sort_by].dropna(), horizontal=True)

elif page == "Compare two experiments":
    st.header("Compare two experiments")
    col_a, col_b, col_m = st.columns(3)
    a = col_a.selectbox("A", names, index=names.index(reference))
    b = col_b.selectbox("B", names, index=names.index("combined_v2") if "combined_v2" in names else 0)
    metric = col_m.selectbox("Metric", metrics, index=metrics.index("recall@5") if "recall@5" in metrics else 0)
    qa, qb = per_question(scores, a, metric), per_question(scores, b, metric)
    both = pd.concat({"A": qa, "B": qb}, axis=1).dropna()
    if both.empty:
        st.info(f"No question has {metric} in both experiments.")
    else:
        both["Δ"] = both["B"] - both["A"]
        st.metric(f"{metric}, mean over {len(both)} questions", f"{both['B'].mean():.3f}", f"{both['Δ'].mean():+.3f} vs A")
        by_type = both.groupby("question_type")[["A", "B", "Δ"]].mean().sort_values("Δ")
        st.subheader("By question type")
        st.dataframe(by_type.style.format("{:.3f}"), width="stretch")
        st.bar_chart(by_type["Δ"], horizontal=True)
        changed = both[both["Δ"].abs() > 1e-9].reset_index()
        changed["question"] = changed["question_id"].map(lambda q: questions[q]["question"])
        for title, rows in [("Improved in B", changed[changed["Δ"] > 0].sort_values("Δ", ascending=False)),
                            ("Degraded in B", changed[changed["Δ"] < 0].sort_values("Δ"))]:
            st.subheader(f"{title} ({len(rows)})")
            st.dataframe(rows[["question_id", "question_type", "question", "A", "B", "Δ"]], hide_index=True,
                         width="stretch")

else:
    st.header("Question detail")
    ids = sorted(questions)
    label = lambda q: f"{q} [{questions[q]['type']}] {questions[q]['question']}"
    qid = st.selectbox("Question", ids, format_func=label, index=ids.index("Q013") if "Q013" in ids else 0)
    question = questions[qid]
    st.markdown(f"**{question['question']}**  \nType: `{question['type']}` · expected answer: *{question['answer']}*")
    if golden[qid]["evidence"]:
        st.dataframe(pd.DataFrame(golden[qid]["evidence"]), hide_index=True, width="stretch")

    with db.connect(read_only=True) as conn:
        answers = db.question_answers(conn, qid)
        question_scores = pd.DataFrame(db.question_scores(conn, qid), columns=["experiment_id", "metric", "value", "reason"])
    latest = {name: (eid, answer, chunks, latency) for eid, name, answer, chunks, latency in answers}  # last run wins
    default = [n for n in [reference, "combined_v2"] if n in latest]
    shown = st.multiselect("Experiments (most recent run of each)", list(latest), default=default)

    for column, name in zip(st.columns(max(len(shown), 1)), shown):
        eid, answer, chunks, latency = latest[name]
        with column:
            st.subheader(name)
            st.caption(f"run {eid} · {latency:.0f} ms")
            st.markdown(f"> {answer}" if answer else "*retrieval only: no answer generated*")
            rows = question_scores[question_scores["experiment_id"] == eid]
            st.dataframe(rows[["metric", "value", "reason"]], hide_index=True, width="stretch")
            st.dataframe(pd.DataFrame([{
                "rank": i, "document": c["document"], "chunk": c["chunk_index"],
                "relevant": "✓" if judge_chunk(c, golden[qid]) else "",  # right document and every snippet
                "rerank": c.get("rerank_score"), "search": c.get("search_score"),
                "text": " / ".join(c["text"].splitlines()[1:])[:200],
            } for i, c in enumerate(chunks, 1)]), hide_index=True, width="stretch")
