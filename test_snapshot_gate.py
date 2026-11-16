"""Unit and smoke tests for snapshot_gate.py.  Run: python test_snapshot_gate.py"""
import math, os, tempfile, unittest
import numpy as np
import torch
import snapshot_gate as sg


def A_matrix(T):
    """Dense A for one channel: [H*W, C*H*W]."""
    C, H, W = T.shape
    A = torch.zeros(H * W, C * H * W, dtype=torch.float64)
    for k in range(C):
        A[torch.arange(H * W), k * H * W + torch.arange(H * W)] = T[k].flatten().double()
    return A


class TestCodes(unittest.TestCase):
    def test_orthonormal_rows(self):
        for kind in ("avg", "dcwalsh", "cacti", "randpm"):
            for C in (2, 4, 8, 16):
                T = sg.fixed_code(kind, C, 6, 7)
                A = A_matrix(T)
                err = (A @ A.T - torch.eye(A.shape[0], dtype=torch.float64)).abs().max()
                self.assertLess(float(err), 1e-6, f"{kind} C={C}")
        T = sg.LearnedCode(8)(9, 10).detach()
        self.assertLess(float((T.norm(dim=0) - 1).abs().max()), 1e-6)

    def test_static_scene_is_scaled_image(self):
        C, a = 8, 0.7071
        img = torch.randn(1, 1, 3, 12, 12)
        f = img.expand(1, C, 3, 12, 12)
        g = sg.snapshot(f, sg.fixed_code("dcwalsh", C, 12, 12, a))
        self.assertTrue(torch.allclose(g, a * math.sqrt(C) * img[:, 0], atol=1e-5))

    def test_dcwalsh_uses_all_temporal_patterns(self):
        C = 8
        T = sg.fixed_code("dcwalsh", C, 8, 8)
        Wh = sg.walsh(C)
        detail = T - 0.7071 * Wh[0].view(C, 1, 1)
        rows = {int((Wh[1:] @ detail[:, y, x]).abs().argmax()) for y in range(8) for x in range(8)}
        self.assertEqual(rows, set(range(C - 1)))

    def test_projection_is_exact(self):
        torch.manual_seed(0)
        for kind in ("dcwalsh", "cacti", "randpm"):
            T = sg.fixed_code(kind, 4, 10, 10)
            f = torch.randn(2, 4, 3, 10, 10)
            g = sg.snapshot(f, T)
            fhat = sg.project(torch.randn_like(f), g, T)
            self.assertLess(float((sg.snapshot(fhat, T) - g).abs().max()), 1e-5)
            # projecting the truth leaves it unchanged
            self.assertLess(float((sg.project(f, g, T) - f).abs().max()), 1e-5)

    def test_noise_maps_to_unit_gaussian(self):
        torch.manual_seed(1)
        for kind in ("dcwalsh", "cacti", "randpm", "avg"):
            T = sg.fixed_code(kind, 8, 64, 64)
            e = torch.randn(64, 8, 4, 64, 64)
            ge = sg.snapshot(e, T)
            self.assertLess(abs(float(ge.mean())), 0.01)
            self.assertLess(abs(float(ge.std()) - 1.0), 0.01)

    def test_diffusion_commutes(self):
        """A(alpha f0 + sigma eps) = alpha A f0 + sigma A eps, with A eps ~ N(0, I)."""
        torch.manual_seed(2)
        T = sg.fixed_code("dcwalsh", 4, 16, 16)
        f0, eps = torch.randn(3, 4, 2, 16, 16), torch.randn(3, 4, 2, 16, 16)
        al, si = 0.6, 0.8
        lhs = sg.snapshot(al * f0 + si * eps, T)
        rhs = al * sg.snapshot(f0, T) + si * sg.snapshot(eps, T)
        self.assertTrue(torch.allclose(lhs, rhs, atol=1e-5))


class TestModels(unittest.TestCase):
    def test_shapes_and_projection_in_models(self):
        f = torch.randn(2, 4, 3, 13, 17)
        for kind in sg.ARMS:
            m = sg.make_arm(kind, 4, 3, 8)
            out = m(f)
            self.assertEqual(out.shape, f.shape, kind)
            if kind != "learned_ae":
                g, T = m.encode(f)
                self.assertLess(float((sg.snapshot(out, T) - g).abs().max()), 1e-4, kind)

    def test_every_arm_learns(self):
        class A: pass
        args = A(); args.data = "synthetic"; args.channels = 3; args.size = 16; args.max_speed = 2.0
        src = sg.ClipSource(args)
        g = torch.Generator().manual_seed(0)
        for kind in sg.ARMS:
            torch.manual_seed(0)
            m = sg.make_arm(kind, 4, 3, 8)
            opt = torch.optim.Adam(m.parameters(), lr=3e-3)
            fixed = src.batch(8, 4, g)
            first = None
            for _ in range(40):
                loss = torch.nn.functional.mse_loss(m(fixed), fixed)
                first = first if first is not None else float(loss)
                opt.zero_grad(); loss.backward(); opt.step()
            self.assertLess(float(loss), first, kind)


class TestData(unittest.TestCase):
    def test_synthetic_motion_scales_with_speed(self):
        g = torch.Generator().manual_seed(0)
        slow = torch.stack([sg.synthetic_clip(4, 32, 3, 0.0, g) for _ in range(8)])
        fast = torch.stack([sg.synthetic_clip(4, 32, 3, 4.0, g) for _ in range(8)])
        self.assertLess(float(sg.motion(slow).mean()), 1e-6)
        self.assertGreater(float(sg.motion(fast).mean()), 1e-3)

    def test_npy_source_normalises(self):
        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(0)
            for i in range(12):
                np.save(os.path.join(d, f"clip{i:02d}.npy"),
                        (rng.standard_normal((9, 16, 20, 24)) * 3 + 5).astype(np.float16))
            class A: pass
            args = A(); args.data = "npy"; args.path = d; args.size = 16; args.channels = 0
            src = sg.ClipSource(args)
            b = src.batch(32, 4, torch.Generator().manual_seed(0))
            self.assertEqual(tuple(b.shape), (32, 4, 16, 16, 16))
            self.assertLess(abs(float(b.mean())), 0.1)
            self.assertLess(abs(float(b.std()) - 1), 0.1)


class TestDecision(unittest.TestCase):
    def test_rule(self):
        mk = lambda s, hm: dict(snr_db=s, high_motion_motion_snr_db=hm)
        res = {4: {"avg": mk(20, 5), "dcwalsh": mk(24, 12), "learned_ae": mk(24.5, 13)}}
        d = sg.decide(res, 4)
        self.assertTrue(d["go"]); self.assertEqual(d["best_coded_arm"], "dcwalsh")
        res[4]["learned_ae"] = mk(27, 15)
        self.assertFalse(sg.decide(res, 4)["go"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
