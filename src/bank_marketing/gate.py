"""Model promotion gate.

Encapsulates the single rule that decides whether a freshly trained candidate model
should replace the current production model. The goal of the gate is to *never let the
chosen metric regress*: a candidate is promoted only when it is at least as good as the
current production model (within a configurable tolerance), or when no production model
exists yet.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionDecision:
    """Outcome of the promotion gate."""

    promote: bool
    reason: str


def decide_promotion(
    candidate_metrics: dict[str, float],
    current_metrics: dict[str, float] | None,
    *,
    metric: str = "roc_auc",
    tolerance: float = 0.0,
) -> PromotionDecision:
    """Decide whether to promote the candidate over the current production model.

    Args:
        candidate_metrics: Metrics of the newly trained model.
        current_metrics: Metrics of the current production model, or ``None`` if there is
            no production model yet.
        metric: Which metric to compare on.
        tolerance: How much worse than production the candidate may be and still pass
            (``0.0`` means it must be at least as good).
    """
    candidate = candidate_metrics[metric]
    if current_metrics is None:
        return PromotionDecision(
            promote=True,
            reason=f"No production model yet; promoting candidate ({metric}={candidate:.4f}).",
        )

    current = current_metrics[metric]
    if candidate >= current - tolerance:
        return PromotionDecision(
            promote=True,
            reason=(
                f"Candidate {metric}={candidate:.4f} >= production {current:.4f} "
                f"- tolerance {tolerance:.4f}; promoting."
            ),
        )
    return PromotionDecision(
        promote=False,
        reason=(
            f"Candidate {metric}={candidate:.4f} < production {current:.4f} "
            f"- tolerance {tolerance:.4f}; keeping production."
        ),
    )
