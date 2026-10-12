import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from prognostics.full_agent_benchmark import ConcurrentBudget, select_cases


class FullAgentStudyTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for equipment in range(10):
            for index in range(12):
                image = [index in (0, 1, 2, 3, 10, 11)]
                history = [0] * 13
                history[10] = index not in (8, 9)
                history[11] = index in (6, 7)
                history[12] = 30 if index in (10, 11) else 0
                event = history.copy()
                if history[11]:
                    event[10] = 0
                rows.append(
                    {
                        "equipment": equipment,
                        "row": index,
                        "image": image,
                        "numeric": [30 if index in (2, 3) else 20],
                        "history": history,
                        "event_history": event,
                    }
                )
        return rows

    def test_selection_distinct_pairs_and_equipment_without_targets(self):
        selected = select_cases(self.fixture())
        self.assertEqual(len(selected), 24)
        self.assertEqual(len({(r["equipment"], r["row"]) for _, r in selected}), 24)
        for name in {s for s, _ in selected}:
            self.assertEqual(len({r["equipment"] for s, r in selected if s == name}), 4)
        self.assertEqual(selected, select_cases(self.fixture()))

    def test_insufficient_equipment_fails_before_api(self):
        with self.assertRaises(ValueError):
            select_cases([r for r in self.fixture() if r["equipment"] < 3])

    def test_parallel_reservations_released_and_usage_measured(self):
        budget = ConcurrentBudget(100000)

        def request(_):
            budget.admit([], [], 1000)
            budget.record(SimpleNamespace(usage=SimpleNamespace(total_tokens=300)))

        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(request, range(20)))
        self.assertEqual(budget.requests, 20)
        self.assertEqual(budget.used, 6000)
        self.assertEqual(budget.pending, 0)

    def test_outstanding_reservation_blocks_over_admission(self):
        budget = ConcurrentBudget(1800)
        budget.admit([], [], 1000)
        with self.assertRaises(ValueError):
            budget.admit([], [], 1000)
        self.assertEqual(budget.requests, 1)


if __name__ == "__main__":
    unittest.main()
