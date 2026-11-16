from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from scripts import patch_tubeletgraph, tubelet_speedups as speedups

try:
    import torch
except ImportError:  # pragma: no cover - torch is only in the GPU image
    torch = None


def _backbone(seed: float):
    seed = float(seed)
    fpn = [torch.full((1, 2, 4, 4), seed), torch.full((1, 3, 2, 2), seed + 1)]
    pos = [torch.arange(32.0).reshape(1, 2, 4, 4), torch.arange(12.0).reshape(1, 3, 2, 2)]
    return {"vision_features": fpn[-1], "backbone_fpn": fpn, "vision_pos_enc": pos}


@unittest.skipIf(torch is None, "torch not installed")
class FeatureCacheTests(unittest.TestCase):
    def test_round_trip_returns_identical_features(self) -> None:
        cache = speedups.FeatureCache(object(), gpu_bytes=10**9, cpu_bytes=0)
        self.assertIsNone(cache.get(3, "cpu"))
        original = _backbone(5.0)
        cache.put(3, original)
        restored = cache.get(3, "cpu")
        for a, b in zip(original["backbone_fpn"], restored["backbone_fpn"]):
            self.assertTrue(torch.equal(a, b))
        for a, b in zip(original["vision_pos_enc"], restored["vision_pos_enc"]):
            self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(restored["vision_features"], original["backbone_fpn"][-1]))
        self.assertEqual((cache.hits, cache.misses), (1, 1))

    def test_budget_spills_to_cpu_then_skips(self) -> None:
        one_frame = sum(t.numel() * t.element_size() for t in _backbone(0.0)["backbone_fpn"])
        cache = speedups.FeatureCache(object(), gpu_bytes=one_frame, cpu_bytes=one_frame)
        for frame in range(3):
            cache.put(frame, _backbone(float(frame)))
        self.assertEqual(sorted(cache.entries), [0, 1])
        self.assertEqual(cache.uncached, 1)
        self.assertIsNone(cache.get(2, "cpu"))
        self.assertTrue(torch.equal(cache.get(1, "cpu")["backbone_fpn"][0], _backbone(1.0)["backbone_fpn"][0]))

    def test_differing_positional_encodings_are_kept_per_frame(self) -> None:
        cache = speedups.FeatureCache(object(), gpu_bytes=10**9, cpu_bytes=0)
        first, second = _backbone(0.0), _backbone(1.0)
        second["vision_pos_enc"] = [t + 1 for t in second["vision_pos_enc"]]
        cache.put(0, first)
        cache.put(1, second)
        self.assertTrue(torch.equal(cache.get(0, "cpu")["vision_pos_enc"][0], first["vision_pos_enc"][0]))
        self.assertTrue(torch.equal(cache.get(1, "cpu")["vision_pos_enc"][0], second["vision_pos_enc"][0]))


class SharingTests(unittest.TestCase):
    def setUp(self) -> None:
        speedups._PREDICTORS.clear()
        speedups._FEATURE_CACHES.clear()
        speedups._FRAMES.clear()

    def test_flags_off_builds_every_time(self) -> None:
        builder = mock.Mock(side_effect=lambda *a, **k: object())
        with mock.patch.dict(os.environ, {}, clear=True):
            first = speedups.build_predictor(builder, "cfg", "w.pt", "cuda")
            second = speedups.build_predictor(builder, "cfg", "w.pt", "cuda")
            state: dict = {"num_frames": 4}
            speedups.attach_feature_cache(first, state, "/v")
        self.assertIsNot(first, second)
        self.assertNotIn("_moscato_feature_cache", state)

    def test_shared_predictor_and_cache(self) -> None:
        builder = mock.Mock(side_effect=lambda *a, **k: object())
        env = {speedups.SHARE_ENV: "1", speedups.CACHE_ENV: "1"}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(speedups, "_install_frame_cache"):
            first = speedups.build_predictor(builder, "cfg", "w.pt", "cuda")
            second = speedups.build_predictor(builder, "cfg", "w.pt", "cuda")
            a, b = {"num_frames": 4}, {"num_frames": 4}
            speedups.attach_feature_cache(first, a, "/v")
            speedups.attach_feature_cache(second, b, "/v/")
        self.assertIs(first, second)
        builder.assert_called_once()
        self.assertIs(a["_moscato_feature_cache"], b["_moscato_feature_cache"])

    def test_frame_loader_is_memoized_per_video(self) -> None:
        loads = []

        def fake_loader(video_path, image_size, offload_video_to_cpu,
                        async_loading_frames=False, compute_device="cuda"):
            loads.append(video_path)
            return (object(), 10, 20)

        fake_module = types.SimpleNamespace(load_video_frames=fake_loader)
        fake_package = types.ModuleType("sam2")
        fake_package.sam2_video_predictor = fake_module
        fake_torch = types.SimpleNamespace(device=lambda name: name)
        modules = {"sam2": fake_package, "sam2.sam2_video_predictor": fake_module}
        if torch is None:
            modules["torch"] = fake_torch
        with mock.patch.dict(sys.modules, modules):
            speedups._install_frame_cache()
            speedups._install_frame_cache()  # idempotent
            loader = fake_module.load_video_frames
            first = loader(video_path="/v", image_size=1024, offload_video_to_cpu=False,
                           async_loading_frames=False, compute_device="cuda")
            second = loader(video_path="/v", image_size=1024, offload_video_to_cpu=False,
                            async_loading_frames=False, compute_device="cuda")
            loader(video_path="/w", image_size=1024, offload_video_to_cpu=False,
                   async_loading_frames=False, compute_device="cuda")
        self.assertIs(first, second)
        self.assertEqual(loads, ["/v", "/w"])
        speedups._FRAMES.clear()

    def test_lazy_frame_loader_is_shared_and_does_not_call_eager_loader(self) -> None:
        eager = mock.Mock()
        eager._moscato_cached = False
        fake_module = types.SimpleNamespace(load_video_frames=eager)
        fake_package = types.ModuleType("sam2")
        fake_package.sam2_video_predictor = fake_module
        modules = {"sam2": fake_package, "sam2.sam2_video_predictor": fake_module}
        if torch is None:
            modules["torch"] = types.SimpleNamespace(device=lambda name: name)
        env = {speedups.SHARE_ENV: "1", speedups.LAZY_CPU_ENV: "1"}
        with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(speedups, "_lazy_cpu_frames", return_value=(object(), 10, 20)) as lazy:
            speedups._install_frame_cache()
            first = fake_module.load_video_frames("/v", 1024, True)
            second = fake_module.load_video_frames("/v", 1024, True)
            self.assertIs(first, second)
            lazy.assert_called_once()
            eager.assert_not_called()
            with self.assertRaisesRegex(ValueError, "offload_video_to_cpu"):
                fake_module.load_video_frames("/v", 1024, False)

    def test_frame_offload_is_opt_in(self) -> None:
        predictor = mock.Mock()
        with mock.patch.dict(os.environ, {}, clear=True):
            speedups.initialize_state(predictor, "/v")
        predictor.init_state.assert_called_once_with(video_path="/v")
        predictor.reset_mock()
        with mock.patch.dict(os.environ, {speedups.LAZY_CPU_ENV: "1"}, clear=True), \
             mock.patch.object(speedups, "_install_frame_cache"):
            speedups.initialize_state(predictor, "/v")
        predictor.init_state.assert_called_once_with(video_path="/v", offload_video_to_cpu=True)


@unittest.skipIf(torch is None, "torch not installed")
class LazyCPUFrameTests(unittest.TestCase):
    def setUp(self):
        self.loads = []
        self.raw = [torch.linspace(0.001 + i * 0.01, 0.91, 48, dtype=torch.float64).reshape(3, 4, 4)
                    for i in range(3)]
        def load(path, image_size):
            index = int(path)
            self.loads.append(index)
            return self.raw[index].clone(), 6, 8
        self.loader = mock.Mock(side_effect=load)
        fake_misc = types.ModuleType("sam2.utils.misc")
        fake_misc._load_img_as_tensor = self.loader
        self.modules = mock.patch.dict(sys.modules, {
            "sam2": types.ModuleType("sam2"), "sam2.utils": types.ModuleType("sam2.utils"),
            "sam2.utils.misc": fake_misc,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_float32_pixels_match_eager_normalization_exactly(self):
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
        dense = torch.zeros(3, 3, 4, 4, dtype=torch.float32)
        for index, frame in enumerate(self.raw):
            dense[index] = frame  # Pinned eager loader casts before normalizing.
        dense -= mean
        dense /= std
        frames = speedups.LazyCPUFrames(["0", "1", "2"], 4, cache_size=2)
        for index in range(3):
            self.assertEqual(frames[index].dtype, torch.float32)
            self.assertEqual(frames[index].device.type, "cpu")
            self.assertTrue(torch.equal(frames[index], dense[index]))
        self.assertFalse(torch.equal(((self.raw[0] - mean) / std).float(), dense[0]))

    def test_lru_is_bounded_and_evicted_frames_are_reloaded(self):
        frames = speedups.LazyCPUFrames(["0", "1", "2"], 4, cache_size=2)
        self.assertEqual(self.loads, [0])
        self.assertEqual(len(frames), 3)
        self.assertEqual((frames.video_height, frames.video_width), (6, 8))
        frames[1]
        first = frames[0]
        frames[2]
        self.assertEqual(list(frames._cache), [0, 2])
        frames[1]
        self.assertEqual(self.loads, [0, 1, 2, 1])
        self.assertEqual(len(frames._cache), 2)
        self.assertTrue(torch.equal(first, frames[0]))
        with self.assertRaises(IndexError):
            frames[3]

    def test_changed_frame_dimensions_are_rejected(self):
        frames = speedups.LazyCPUFrames(["0", "1", "2"], 4)
        self.loader.side_effect = lambda path, image_size: (self.raw[1].clone(), 7, 8)
        with self.assertRaisesRegex(ValueError, "dimensions changed"):
            frames[1]

    def test_empty_sequence_and_nonpositive_cache_are_rejected(self):
        with self.assertRaises(ValueError):
            speedups.LazyCPUFrames([], 4)
        with self.assertRaises(ValueError):
            speedups.LazyCPUFrames(["0"], 4, cache_size=0)


class PatchTests(unittest.TestCase):
    def _fake_repo(self, root: Path) -> None:
        tracker = root / "TubeletGraph" / "tracker"
        tracker.mkdir(parents=True)
        (tracker / "sam2.py").write_text(
            "import torch\n"
            + patch_tubeletgraph.WRAPPER_IMPORT_MARKER
            + "class SAM2:\n    def __init__(self, model_cfg, model_weights, device):\n"
            + patch_tubeletgraph.WRAPPER_BUILD_MARKER
            + "    def initialize(self, **info):\n"
            + patch_tubeletgraph.WRAPPER_INIT_MARKER
        )
        predictor = root / "thirdparty" / "sam2" / "sam2"
        predictor.mkdir(parents=True)
        (predictor / "sam2_video_predictor.py").write_text(
            "def f(self, inference_state, frame_idx, image, device):\n"
            "        if True:\n"
            + patch_tubeletgraph.SAM2_FEATURE_MARKER
        )

    def test_patch_applies_once_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fake_repo(root)
            first = patch_tubeletgraph.patch_speedups(root)
            self.assertTrue(all(changed for _, changed in first))
            second = patch_tubeletgraph.patch_speedups(root)
            self.assertFalse(any(changed for path, changed in second[1:]))
            wrapper = (root / "TubeletGraph/tracker/sam2.py").read_text()
            self.assertIn("_moscato_speedups.build_predictor(", wrapper)
            self.assertIn("_moscato_speedups.attach_feature_cache(", wrapper)
            self.assertIn("_moscato_speedups.initialize_state(", wrapper)
            self.assertTrue((root / "TubeletGraph/tracker/moscato_speedups.py").is_file())
            predictor = (root / "thirdparty/sam2/sam2/sam2_video_predictor.py").read_text()
            self.assertEqual(predictor.count("self.forward_image(image)"), 1)
            self.assertIn("_moscato_cache.put(frame_idx, backbone_out)", predictor)

    def test_previous_frame_cache_patch_upgrades_without_duplicate_attachment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fake_repo(root)
            wrapper = root / "TubeletGraph/tracker/sam2.py"
            wrapper.write_text(wrapper.read_text().replace(
                patch_tubeletgraph.WRAPPER_INIT_MARKER, patch_tubeletgraph.WRAPPER_INIT_LEGACY))
            patch_tubeletgraph.patch_speedups(root)
            source = wrapper.read_text()
            self.assertEqual(source.count("_moscato_speedups.attach_feature_cache("), 1)
            self.assertEqual(source.count("_moscato_speedups.initialize_state("), 1)

    def test_missing_marker_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fake_repo(root)
            (root / "TubeletGraph/tracker/sam2.py").write_text("changed upstream\n")
            with self.assertRaises(SystemExit):
                patch_tubeletgraph.patch_speedups(root)


class CropFormerCacheTests(unittest.TestCase):
    def _helpers(self) -> dict:
        import json
        import os.path as osp
        namespace = {"os": os, "json": json, "osp": osp}
        exec(patch_tubeletgraph.CROPFORMER_HELPERS, namespace)
        return namespace

    def test_round_trip_and_disabled(self) -> None:
        helpers = self._helpers()
        cfg = types.SimpleNamespace(config_path="c.yaml", opts=["MODEL.WEIGHTS", "w.pth"])
        with tempfile.TemporaryDirectory() as tmp:
            frame = Path(tmp) / "0000000.jpg"
            frame.write_bytes(b"jpeg-bytes")
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertIsNone(helpers["_moscato_cf_key"](str(frame), cfg, 0.5))
            env = {"TUBELETGRAPH_CROPFORMER_CACHE_DIR": str(Path(tmp) / "cache")}
            with mock.patch.dict(os.environ, env, clear=True):
                key = helpers["_moscato_cf_key"](str(frame), cfg, 0.5)
                self.assertIsNone(helpers["_moscato_cf_get"](key))
                helpers["_moscato_cf_put"](key, {0: {"counts": "abc", "size": [2, 2]}})
                self.assertEqual(helpers["_moscato_cf_get"](key), {"0": {"counts": "abc", "size": [2, 2]}})
                other = helpers["_moscato_cf_key"](str(frame), cfg, 0.6)
                self.assertNotEqual(key, other)

    def test_patch_inserts_lookup_and_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "TubeletGraph" / "entity_segmentation" / "cropformer.py"
            target.parent.mkdir(parents=True)
            target.write_text(
                patch_tubeletgraph.CROPFORMER_HELPERS_MARKER + "    pass\n"
                + patch_tubeletgraph.CROPFORMER_LOOP_MARKER
                + "            img = read_image(path)\n"
                + patch_tubeletgraph.CROPFORMER_STORE_MARKER
            )
            self.assertTrue(all(c for _, c in patch_tubeletgraph.patch_cropformer_cache(root)))
            self.assertFalse(any(c for _, c in patch_tubeletgraph.patch_cropformer_cache(root)))
            source = target.read_text()
            self.assertEqual(source.count("def _moscato_cf_key"), 1)
            self.assertLess(source.index("_moscato_cf_get(_moscato_key)"), source.index("img = read_image"))
            self.assertIn("_moscato_cf_put(_moscato_key, out[len(out) - 1])", source)


if __name__ == "__main__":
    unittest.main()
