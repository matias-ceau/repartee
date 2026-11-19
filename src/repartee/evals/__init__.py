"""Measuring the mirror: the blind voice test, style distance and recall.

Every result is saved as JSON under the home's ``evals/`` folder, never in the code repository.
"""

from repartee.evals.recall import FactGrade, QuestionResult, RecallResult, run_recall
from repartee.evals.style_distance import style_distance_report
from repartee.evals.voice_blind import BlindPair, BlindTest

__all__ = [
    "BlindPair",
    "BlindTest",
    "FactGrade",
    "QuestionResult",
    "RecallResult",
    "run_recall",
    "style_distance_report",
]
