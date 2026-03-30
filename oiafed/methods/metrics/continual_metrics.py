"""
Continual Learning metrics: accuracy matrix, average accuracy,
forgetting measure, backward/forward transfer.
"""

import numpy as np
from typing import Dict


class ContinualLearningMetrics:
    """Track and compute standard continual-learning evaluation metrics."""

    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks
        # R[i][j] = accuracy on task j after learning task i
        self.R = np.zeros((num_tasks, num_tasks))
        self._updated = np.zeros(num_tasks, dtype=bool)

    def update(self, current_task: int, task_accuracies: Dict[int, float]):
        """Record accuracies on all seen tasks after finishing *current_task*."""
        for task_id, acc in task_accuracies.items():
            self.R[current_task][task_id] = acc
        self._updated[current_task] = True

    def get_all_metrics(self, up_to_task: int) -> Dict[str, float]:
        T = up_to_task + 1  # number of tasks seen so far

        # Average Accuracy: mean of R[T-1][0..T-1]
        average_accuracy = float(np.mean(self.R[T - 1, :T]))

        # Forgetting Measure: mean over tasks j < T-1 of max_{l<=T-2} R[l][j] - R[T-1][j]
        forgetting = 0.0
        if T > 1:
            f_values = []
            for j in range(T - 1):
                best_prev = np.max(self.R[:T - 1, j])
                f_values.append(best_prev - self.R[T - 1, j])
            forgetting = float(np.mean(f_values))

        # Backward Transfer
        bwt = 0.0
        if T > 1:
            bwt_values = [self.R[T - 1, j] - self.R[j, j] for j in range(T - 1)]
            bwt = float(np.mean(bwt_values))

        # Forward Transfer
        fwt = 0.0
        if T > 1:
            fwt_values = [self.R[j - 1, j] for j in range(1, T)]
            fwt = float(np.mean(fwt_values))

        return {
            "average_accuracy": average_accuracy,
            "forgetting_measure": forgetting,
            "backward_transfer": bwt,
            "forward_transfer": fwt,
        }

    def print_accuracy_matrix(self):
        print("\n  Accuracy Matrix R[i][j]:")
        header = "       " + "".join(f"Task {j:>2}  " for j in range(self.num_tasks))
        print(header)
        for i in range(self.num_tasks):
            if not self._updated[i]:
                continue
            row = f"T {i:>2} | " + "  ".join(
                f"{self.R[i][j]:.4f}" if self.R[i][j] > 0 else "  --  "
                for j in range(self.num_tasks)
            )
            print(row)
        print()