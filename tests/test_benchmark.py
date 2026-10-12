import unittest

from prognostics.benchmark import ablated_rows, fold_split, stressed_rows


class BenchmarkTests(unittest.TestCase):
    def test_outer_fold_roles_are_disjoint(self):
        rows = [{"equipment": i} for i in range(1, 51)]
        _, groups = fold_split(rows, list(range(1, 11)), 7)
        self.assertEqual([len(ids) for ids in groups.values()], [25, 5, 10, 10])
        self.assertEqual(len(set(sum(groups.values(), []))), 50)

    def test_dropped_current_image_does_not_survive_in_memory(self):
        rows = [
            {
                "equipment": 1,
                "time": float(i),
                "numeric": [5, 100, 0, 0],
                "image": [1.0] + [float(i) / 10] * 8,
                "history": [0.0] * 13,
                "event_history": [0.0] * 13,
                "text": "stable",
            }
            for i in range(10)
        ]
        altered = stressed_rows(rows, "block_image_outage", 7)
        self.assertEqual(altered[5]["image"], [0.0] * 9)
        self.assertEqual(altered[5]["history"][:9], rows[3]["image"])
        self.assertNotEqual(altered[5]["history"][:9], rows[5]["image"])
        self.assertEqual(altered[5]["history"][9], 2.0)

    def test_full_ablation_changes_all_history_age_inputs(self):
        row = {"history": [1.0] * 13, "event_history": [2.0] * 13}
        altered = ablated_rows([row], "router_without_freshness")[0]
        for key in ["history", "event_history"]:
            self.assertEqual(altered[key][9], 0)
            self.assertEqual(altered[key][12], 0)
        self.assertEqual(row["history"][9], 1.0)

    def test_no_reset_keeps_past_image_and_removes_explicit_flag(self):
        row = {"history": [1.0] * 13, "event_history": [0.0] * 13}
        altered = ablated_rows([row], "router_without_reset")[0]
        self.assertEqual(altered["event_history"][:9], [1.0] * 9)
        self.assertEqual(altered["event_history"][11], 0.0)
