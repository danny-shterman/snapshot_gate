"""Tests for benchmark.py.  Run: python test_benchmark.py"""
import math, unittest
import torch
import benchmark as bm


class TestAnalytic(unittest.TestCase):
    def test_wan14b_numbers(self):
        rows = {r["C"]: r for r in bm.analytic(bm.PRESETS["wan14b_720p"], [1, 4, 8])}
        self.assertEqual(rows[1]["tokens"], 75_600)
        self.assertAlmostEqual(rows[1]["flops_step"] / 1e15, 6.80, places=1)
        self.assertAlmostEqual(rows[8]["speedup"], 17.1, places=1)
        # KV: 40 layers x (K,V) x 5120 x 2 bytes = 819,200 B per token; 3,600 tokens x 4 latent frames/s
        self.assertAlmostEqual(rows[1]["kv_bytes_per_video_second"], 819_200 * 3600 * 4, delta=1)
        self.assertAlmostEqual(rows[8]["kv_reduction"], 8.0, places=6)

    def test_monotone(self):
        rows = bm.analytic(bm.PRESETS["wan1.3b_480p"], [1, 2, 4, 8, 16])
        for a, b in zip(rows, rows[1:]):
            self.assertLess(b["tokens"], a["tokens"])
            self.assertGreater(b["speedup"], a["speedup"])


class TestMeasured(unittest.TestCase):
    def test_streaming_kv_matches_formula(self):
        torch.manual_seed(0)
        m = bm.TinyDiT(64, 2, 4).eval()
        rows = {r["C"]: r for r in bm.measure_streaming(m, [1, 4], 9, 8, 4.0, torch.device("cpu"), torch.float32)}
        for C, r in rows.items():
            tokens = math.ceil(8 / C) * 9
            self.assertEqual(r["cached_tokens"], tokens)
            self.assertEqual(r["kv_bytes"], 2 * 2 * 64 * tokens * 4)        # layers x (K,V) x width x tokens x fp32
        self.assertAlmostEqual(rows[4]["kv_reduction"], 4.0)

    def test_streaming_equals_bidirectional_when_one_chunk(self):
        """Sanity: one chunk with an empty cache is the same computation as a plain forward pass."""
        torch.manual_seed(0)
        m = bm.TinyDiT(32, 2, 2).eval()
        x = torch.randn(1, 12, 16)
        with torch.no_grad():
            a = m(x)
            b = m(x, [dict(k=None, v=None) for _ in m.blocks])
        self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_step_speedup_positive(self):
        m = bm.TinyDiT(64, 2, 4).eval()
        rows = bm.measure_bidirectional(m, [1, 4], 64, 8, torch.device("cpu"), torch.float32, repeats=3)
        self.assertGreater(rows[1]["speedup"], 1.5)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
