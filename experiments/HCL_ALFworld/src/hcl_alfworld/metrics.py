from __future__ import annotations

from typing import Dict, List, Optional


def continual_metrics(
    matrix: List[List[float]], random_baseline: float = 0.0
) -> Dict[str, Optional[float]]:
    """Compute task-IL ACC/BWT/forgetting/FWT from a phase-by-task matrix."""

    if not matrix:
        return {
            "average_accuracy": 0.0,
            "backward_transfer": 0.0,
            "forgetting": 0.0,
            "forward_transfer": None,
        }
    final = matrix[-1]
    # Rows arrive one phase at a time, while each row can already contain
    # scores for future tasks. Retention metrics cover only learned tasks.
    learned_count = min(len(matrix), len(final))
    average_accuracy = sum(final[:learned_count]) / max(learned_count, 1)
    if learned_count <= 1:
        backward_transfer = 0.0
        forgetting = 0.0
    else:
        backward_transfer = sum(final[j] - matrix[j][j] for j in range(learned_count - 1)) / (
            learned_count - 1
        )
        forgetting = sum(
            max(matrix[i][j] for i in range(j, learned_count)) - final[j]
            for j in range(learned_count - 1)
        ) / (learned_count - 1)
    forward_entries = [
        matrix[i][i + 1] - random_baseline
        for i in range(learned_count)
        if i + 1 < len(matrix[i])
    ]
    forward_transfer = (
        sum(forward_entries) / len(forward_entries) if forward_entries else None
    )
    return {
        "average_accuracy": average_accuracy,
        "backward_transfer": backward_transfer,
        "forgetting": forgetting,
        "forward_transfer": forward_transfer,
    }
