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
        # NaN = not evaluated yet; 0.0 = evaluated with zero accuracy
        self.R = np.full((num_tasks, num_tasks), np.nan)
        self._updated = np.zeros(num_tasks, dtype=bool)

    def update(self, current_task: int, task_accuracies: Dict[int, float]):
        """Record accuracies on all seen tasks after finishing *current_task*."""
        for task_id, acc in task_accuracies.items():
            self.R[current_task][task_id] = acc
        self._updated[current_task] = True

    def get_all_metrics(self, up_to_task: int) -> Dict[str, float]:
        T = up_to_task + 1  # number of tasks seen so far

        # Average Accuracy: mean of R[T-1][0..T-1] (ignore NaN)
        row = self.R[T - 1, :T]
        valid = row[~np.isnan(row)]
        average_accuracy = float(np.mean(valid)) if len(valid) > 0 else 0.0

        # Forgetting Measure: mean over tasks j < T-1 of max_{l<=T-2} R[l][j] - R[T-1][j]
        forgetting = 0.0
        if T > 1:
            f_values = []
            for j in range(T - 1):
                col = self.R[:T - 1, j]
                valid_col = col[~np.isnan(col)]
                if len(valid_col) > 0 and not np.isnan(self.R[T - 1, j]):
                    best_prev = np.max(valid_col)
                    f_values.append(best_prev - self.R[T - 1, j])
            forgetting = float(np.mean(f_values)) if f_values else 0.0

        # Backward Transfer
        bwt = 0.0
        if T > 1:
            bwt_values = []
            for j in range(T - 1):
                if not np.isnan(self.R[T - 1, j]) and not np.isnan(self.R[j, j]):
                    bwt_values.append(self.R[T - 1, j] - self.R[j, j])
            bwt = float(np.mean(bwt_values)) if bwt_values else 0.0

        # Forward Transfer
        fwt = 0.0
        if T > 1:
            fwt_values = []
            for j in range(1, T):
                if not np.isnan(self.R[j - 1, j]):
                    fwt_values.append(self.R[j - 1, j])
            fwt = float(np.mean(fwt_values)) if fwt_values else 0.0

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
                f"{self.R[i][j]:.4f}" if not np.isnan(self.R[i][j]) else "  --  "
                for j in range(self.num_tasks)
            )
            print(row)
        print()