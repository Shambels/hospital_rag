"""LLM judge (DeepEval): answer quality on the 26-question sample, scored by a model other than the generator."""
import logging
import os
from collections import defaultdict
from functools import cache

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")  # before importing deepeval: no usage data sent out

from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.models import LocalModel  # noqa: E402
from deepeval.test_case import LLMTestCase, SingleTurnParams  # noqa: E402

from hospital_rag.config import settings  # noqa: E402

logger = logging.getLogger(__name__)

PER_TYPE = 2  # 2 questions per type, out_of_corpus excluded (refusals are measured on all 106): 13 x 2 = 26
METRIC_NAMES = ["contextual_precision", "contextual_recall", "faithfulness", "answer_relevancy", "correctness"]


def judge_sample(questions):
    """The questions the judge scores: the first PER_TYPE ids of each type, so every experiment gets the same ones."""
    by_type = defaultdict(list)
    for question in sorted(questions, key=lambda q: q["id"]):
        if question["type"] != "out_of_corpus" and len(by_type[question["type"]]) < PER_TYPE:
            by_type[question["type"]].append(question)
    return [q for group in by_type.values() for q in group]


@cache
def get_judge(model=None):
    """The judge model on JUDGE_URL. JSON output forced: DeepEval parses every verdict as JSON."""
    return LocalModel(
        model=model or settings.judge_model,
        base_url=settings.judge_url,
        api_key="EMPTY",
        generation_kwargs={"response_format": {"type": "json_object"}},
    )


def make_metrics(judge):
    """The metrics of chapter 04. New objects for each question: a DeepEval metric keeps the state of its last run."""
    return {
        # Retrieval, seen by the judge: are the useful chunks ranked first? Do they hold everything the answer needs?
        "contextual_precision": ContextualPrecisionMetric(model=judge, async_mode=False),
        "contextual_recall": ContextualRecallMetric(model=judge, async_mode=False),
        # Generation: is every claim of the answer backed by the chunks? Does the answer address the question?
        "faithfulness": FaithfulnessMetric(model=judge, penalize_ambiguous_claims=True, async_mode=False),
        "answer_relevancy": AnswerRelevancyMetric(model=judge, async_mode=False),
        # Against the expected answer of the golden dataset
        "correctness": GEval(
            name="correctness",
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
            evaluation_steps=[
                "Compare the facts of the actual output and the expected output.",
                "Penalize any fact that contradicts the expected output.",
                "Missing facts in the actual output should also be penalized.",
            ],
            model=judge,
            async_mode=False,
        ),
    }


def metric_name(name, judge_model):
    """The main judge's scores keep the plain name; another judge's are stored next to them, e.g. "faithfulness (gemma)"."""
    return name if judge_model in (None, settings.judge_model) else f"{name} ({judge_model})"


def judge_answer(question, answer, chunks, judge_model=None):
    """The judge's scores for one answer: {metric: (value, reason)}; value None when the judge failed on a metric."""
    test_case = LLMTestCase(
        input=question["question"],
        actual_output=answer,
        expected_output=question["answer"],
        retrieval_context=[chunk["text"] for chunk in chunks],
    )
    scores = {}
    metrics = make_metrics(get_judge(judge_model))
    assert list(metrics) == METRIC_NAMES  # judge_experiment() relies on the names to resume
    for name, metric in metrics.items():
        try:
            metric.measure(test_case)
            scores[metric_name(name, judge_model)] = (metric.score, metric.reason)
        except Exception as error:  # one malformed verdict must not lose the other metrics, nor the run
            logger.warning("judge failed on %s / %s: %s", question["id"], name, error)
            scores[metric_name(name, judge_model)] = (None, f"judge error: {error}")
    return scores
