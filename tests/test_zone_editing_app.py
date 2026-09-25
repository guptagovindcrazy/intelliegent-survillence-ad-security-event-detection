"""
Integration tests for in-app zone editing: PipelineWorker hot reload / snapshot /
rescaling, the Qt canvas + panel, and the MainWindow edit flow. ultralytics/torch
are stubbed (no weights or GPU needed); PySide6 is required and Qt runs offscreen.
"""

import os
import sys
import tempfile
import time
import types
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    _QT = True
except ImportError:
    _QT = False


def _stub_heavy_deps():
    if "ultralytics" not in sys.modules:
        m = types.ModuleType("ultralytics")
        m.YOLO = m.RTDETR = type("_M", (), {"__init__": lambda self, *a, **k: None})
        sys.modules["ultralytics"] = m
    if "torch" not in sys.modules:
        t = types.ModuleType("torch")
        t.cuda = types.SimpleNamespace(is_available=lambda: False)
        sys.modules["torch"] = t


def _make_video(path, w, h, n):
    import cv2
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
    for _ in range(n):
        wr.write(__import__("numpy").zeros((h, w, 3), dtype="uint8"))
    wr.release()


class _HookLayer:
    """Fake detector: no detections; optionally runs a hook on frame N and can
    sleep per frame so a run stays alive long enough to interact with."""
    hook_at = {}
    delay = 0.0

    def __init__(self, *a, **k):
        self.n = 0

    def process(self, frame):
        self.n += 1
        if self.delay:
            time.sleep(self.delay)
        hook = type(self).hook_at.get(self.n)
        if hook:
            hook()
        return []

    def close(self):
        pass


def _wait(pred, timeout=8.0):
    end = time.time() + timeout
    while time.time() < end:
        QApplication.processEvents()
        if pred():
            return True
        time.sleep(0.01)
    return False


@unittest.skipUnless(_QT, "PySide6 not installed")
class ZoneEditingTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _stub_heavy_deps()
        import desktop_app.pipeline_worker as pw
        import src.layers.detection_tracking as dt
        dt.DetectionTrackingLayer = pw.DetectionTrackingLayer = _HookLayer
        cls.pw = pw
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.clip = os.path.join(tempfile.gettempdir(), "_zone_edit_clip.mp4")
        _make_video(cls.clip, 320, 240, 200)
        cls.big_clip = os.path.join(tempfile.gettempdir(), "_zone_edit_big.mp4")
        _make_video(cls.big_clip, 1920, 1080, 4)

    def setUp(self):
        _HookLayer.hook_at, _HookLayer.delay = {}, 0.0
        f = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
        f.close()
        os.unlink(f.name)  # start with "no zones file"
        self.zones_path = f.name
        self.addCleanup(lambda: os.path.exists(self.zones_path) and os.unlink(self.zones_path))


class TestWorkerZoneHandling(ZoneEditingTestBase):
    def _cfg(self, ref=None):
        from src.config import Zone, ZoneConfig
        return ZoneConfig([Zone("Door", [(100, 100), (200, 100), (200, 200), (100, 200)], "high")], [], ref)

    def test_snapshot_is_clean_and_at_processing_resolution(self):
        w = self.pw.PipelineWorker(self.big_clip, zone_config=self._cfg())
        shots = []
        w.snapshot_ready.connect(shots.append)
        _HookLayer.hook_at = {2: w.request_snapshot}
        w.run()
        self.assertEqual(len(shots), 1, "exactly one snapshot per request")
        self.assertEqual(shots[0].shape[:2], (720, 1280))       # 1080p capped at the processing size
        self.assertEqual(int(shots[0].sum()), 0)                # no zone overlay was drawn on it

    def test_hot_update_swaps_zones_between_frames(self):
        from src.config import Zone, ZoneConfig
        w = self.pw.PipelineWorker(self.clip, zone_config=self._cfg())
        statuses = []
        w.status_changed.connect(statuses.append)
        new = ZoneConfig([Zone("A", [(0, 0), (50, 0), (50, 50)]), Zone("B", [(0, 0), (80, 0), (80, 80)])], [])
        _HookLayer.hook_at = {3: lambda: w.update_zones(new)}
        summaries = []
        w.finished_run.connect(summaries.append)
        w.run()
        self.assertTrue(any("Zones updated" in s and "2 zone(s)" in s for s in statuses), statuses)
        self.assertEqual([z.name for z in summaries[0].zones], ["A", "B"])

    def test_invalid_hot_update_is_rejected_and_run_continues(self):
        from src.config import Zone, ZoneConfig
        w = self.pw.PipelineWorker(self.clip, zone_config=self._cfg())
        statuses, summaries = [], []
        w.status_changed.connect(statuses.append)
        w.finished_run.connect(summaries.append)
        bad = ZoneConfig([Zone("Bad", [(0, 0), (1, 1)])], [])  # 2-point polygon
        _HookLayer.hook_at = {3: lambda: w.update_zones(bad)}
        w.run()
        self.assertTrue(any("rejected" in s for s in statuses))
        self.assertIsNone(summaries[0].error)
        self.assertEqual([z.name for z in summaries[0].zones], ["Door"])

    def test_reference_size_is_rescaled_to_the_processed_frame(self):
        # drawn on a 640x480 frame, run on a 320x240 video -> coordinates halve
        w = self.pw.PipelineWorker(self.clip, zone_config=self._cfg(ref=(640, 480)))
        summaries = []
        w.finished_run.connect(summaries.append)
        w.run()
        self.assertEqual(summaries[0].zones[0].polygon[0], (50, 50))

    def test_fresh_yaml_is_read_when_no_config_is_injected(self):
        from src.zone_editor import ZoneEditorModel
        import src.config as cfg_mod
        m = ZoneEditorModel()
        m.rebase((320, 240))
        for p in [(10, 10), (60, 10), (60, 60)]:
            m.add_point(p)
        m.commit("FromDisk")
        m.save(self.zones_path)
        real = cfg_mod.ZONES_YAML_PATH
        orig = self.pw.load_zone_config
        self.pw.load_zone_config = lambda: orig(self.zones_path)
        self.addCleanup(setattr, self.pw, "load_zone_config", orig)
        w = self.pw.PipelineWorker(self.clip)
        summaries = []
        w.finished_run.connect(summaries.append)
        w.run()
        self.assertEqual([z.name for z in summaries[0].zones], ["FromDisk"])
        self.assertEqual(real, cfg_mod.ZONES_YAML_PATH)


class TestFrameStride(ZoneEditingTestBase):
    def _run(self, stride):
        from src.config import DET_CFG
        old = DET_CFG.frame_stride
        DET_CFG.frame_stride = stride
        self.addCleanup(setattr, DET_CFG, "frame_stride", old)
        calls = []
        orig_process = _HookLayer.process
        _HookLayer.process = lambda self_, frame: (calls.append(1), [])[1]
        self.addCleanup(setattr, _HookLayer, "process", orig_process)
        w = self.pw.PipelineWorker(self.clip)
        summaries, statuses = [], []
        w.finished_run.connect(summaries.append)
        w.status_changed.connect(statuses.append)
        w.run()
        return len(calls), summaries[0], statuses

    def test_default_stride_analyses_every_frame(self):
        n, summary, _ = self._run(1)
        self.assertEqual(n, 200)
        self.assertEqual(summary.frames_processed, 200)

    def test_stride_three_analyses_every_third_frame(self):
        n, summary, statuses = self._run(3)
        self.assertEqual(n, 67)                       # frames 1, 4, 7, ... of 200
        self.assertEqual(summary.frames_processed, 67)
        self.assertIsNone(summary.error)

    def test_invalid_stride_is_rejected_by_config(self):
        from src.config import DetectionConfig
        with self.assertRaises(ValueError):
            DetectionConfig(frame_stride=0)
        with self.assertRaises(ValueError):
            DetectionConfig(cpu_threads=-1)


class TestYouTubeSources(ZoneEditingTestBase):
    """A YouTube link is resolved to a media address before OpenCV opens it (yt-dlp faked)."""

    def _install_fake(self, info=None, error=None):
        mod = types.ModuleType("yt_dlp")

        class YoutubeDL:
            def __init__(self, options): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def extract_info(self, url, download=False):
                if error:
                    raise error
                return info

        mod.YoutubeDL = YoutubeDL
        saved = sys.modules.get("yt_dlp")
        sys.modules["yt_dlp"] = mod
        self.addCleanup(lambda: sys.modules.pop("yt_dlp", None) if saved is None else sys.modules.__setitem__("yt_dlp", saved))

    def test_main_module_uses_the_same_live_or_recorded_rule(self):
        import src.main as main_mod
        self.assertFalse(main_mod.is_live_source("https://host/clip.mp4"))
        self.assertTrue(main_mod.is_live_source("rtsp://cam/stream"))

    def test_youtube_link_runs_through_the_pipeline_as_recorded_video(self):
        self._install_fake({"url": self.clip, "is_live": False})
        w = self.pw.PipelineWorker("https://youtu.be/abc123")
        summaries = []
        w.finished_run.connect(summaries.append)
        w.run()
        self.assertIsNone(summaries[0].error)
        self.assertFalse(summaries[0].is_live, "a normal YouTube video uses its own timeline")
        self.assertEqual(summaries[0].frames_processed, 200)
        self.assertEqual(summaries[0].source, "https://youtu.be/abc123")   # reports show what the user typed

    def test_unresolvable_link_is_reported_as_a_source_problem(self):
        self._install_fake(error=RuntimeError("Private video"))
        w = self.pw.PipelineWorker("https://youtu.be/abc123")
        summaries = []
        w.finished_run.connect(summaries.append)
        w.run()
        self.assertEqual(summaries[0].error_category, "source_open")
        self.assertIn("Private video", summaries[0].error)

    def test_snapshot_worker_resolves_links_too(self):
        from desktop_app.snapshot_worker import SnapshotWorker
        self._install_fake({"url": self.clip, "is_live": False})
        shots, fails = [], []
        sw = SnapshotWorker("https://youtu.be/abc123")
        sw.snapshot_ready.connect(shots.append)
        sw.failed.connect(fails.append)
        sw.run()
        self.assertEqual((len(shots), fails), (1, []))
        self.assertEqual(shots[0].shape[:2], (240, 320))


class TestAlertReportTime(ZoneEditingTestBase):
    def test_live_alert_time_is_relative_to_run_start(self):
        t = self.pw._alert_report_time(1_789_000_020.8, True, 1_789_000_000.0)
        self.assertAlmostEqual(t, 20.8, places=3)

    def test_file_alert_time_is_the_video_timeline_unchanged(self):
        self.assertEqual(self.pw._alert_report_time(12.5, False, 1_789_000_000.0), 12.5)


class TestCanvasAndPanel(ZoneEditingTestBase):
    def _panel(self):
        from desktop_app.zone_editor_panel import ZoneEditorPanel
        import numpy as np
        p = ZoneEditorPanel(path=self.zones_path)
        p.resize(1000, 600)
        p.show()
        p.load(np.zeros((240, 320, 3), dtype=np.uint8))
        p.zone_radio.setChecked(True)   # most tests below click polygon points; box tests pick the box radio
        return p

    def _click(self, panel, img_pt):
        tf = panel.canvas.transform()
        wx, wy = tf.to_widget(*img_pt)
        QTest.mouseClick(panel.canvas, Qt.MouseButton.LeftButton, pos=QPoint(int(round(wx)), int(round(wy))))

    def test_clicks_become_image_coordinates_and_zone_is_saved_with_reference_size(self):
        p = self._panel()
        p._ask_name = lambda default, error="": "Vault"
        saved = []
        p.saved.connect(saved.append)
        for pt in [(20, 20), (200, 30), (180, 200)]:
            self._click(p, pt)
        self.assertEqual(p.model.pending, [(20, 20), (200, 30), (180, 200)])
        QTest.mouseDClick(p.canvas, Qt.MouseButton.LeftButton, pos=QPoint(*map(int, p.canvas.transform().to_widget(180, 200))))
        self.assertEqual([z.name for z in p.model.zones], ["Vault"])
        self.assertTrue(p.save())
        self.assertEqual(saved[0].reference_size, (320, 240))
        from src.config import load_zone_config
        self.assertEqual(load_zone_config(self.zones_path).zones[0].polygon[0], (20, 20))

    def _drag_box(self, p, a, b, modifiers=Qt.KeyboardModifier.NoModifier):
        QTest.mousePress(p.canvas, Qt.MouseButton.LeftButton, modifiers, pos=self._w(p, a))
        QTest.mouseMove(p.canvas, self._w(p, ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)))
        QTest.mouseMove(p.canvas, self._w(p, b))
        QTest.mouseRelease(p.canvas, Qt.MouseButton.LeftButton, modifiers, pos=self._w(p, b))

    def test_box_is_the_default_way_to_draw_a_zone(self):
        from desktop_app.zone_editor_panel import ZoneEditorPanel
        fresh = ZoneEditorPanel(path=self.zones_path)
        self.assertTrue(fresh.box_radio.isChecked())

    def test_dragging_a_box_makes_a_named_rectangular_zone(self):
        p = self._panel()
        p.box_radio.setChecked(True)
        p._ask_name = lambda default, error="": "Server room"
        self._drag_box(p, (40, 30), (220, 160))
        self.assertEqual([z.name for z in p.model.zones], ["Server room"])
        self.assertEqual(p.model.zones[0].polygon, [(40, 30), (220, 30), (220, 160), (40, 160)])
        self.assertEqual(p.model.pending, [])

    def test_a_box_can_be_dragged_in_any_direction(self):
        p = self._panel()
        p.box_radio.setChecked(True)
        p._ask_name = lambda default, error="": "Up-left"
        self._drag_box(p, (220, 160), (40, 30))       # bottom-right to top-left
        self.assertEqual(p.model.zones[0].polygon, [(40, 30), (220, 30), (220, 160), (40, 160)])

    def test_a_stray_click_does_not_make_a_box(self):
        p = self._panel()
        p.box_radio.setChecked(True)
        asked = []
        p._ask_name = lambda default, error="": asked.append(1) or "X"
        self._drag_box(p, (100, 100), (102, 103))     # smaller than the minimum drag
        self.assertEqual((p.model.zones, p.model.pending, asked), ([], [], []))

    def test_cancelling_the_name_discards_the_box(self):
        p = self._panel()
        p.box_radio.setChecked(True)
        p._ask_name = lambda default, error="": None
        self._drag_box(p, (40, 30), (220, 160))
        self.assertEqual((p.model.zones, p.model.pending), ([], []))

    def test_escape_during_a_box_drag_cancels_it(self):
        p = self._panel()
        p.box_radio.setChecked(True)
        asked = []
        p._ask_name = lambda default, error="": asked.append(1) or "X"
        QTest.mousePress(p.canvas, Qt.MouseButton.LeftButton, pos=self._w(p, (40, 30)))
        QTest.mouseMove(p.canvas, self._w(p, (200, 150)))
        QTest.keyClick(p.canvas, Qt.Key.Key_Escape)
        QTest.mouseRelease(p.canvas, Qt.MouseButton.LeftButton, pos=self._w(p, (200, 150)))
        self.assertEqual((p.model.zones, asked), ([], []))

    def test_dragging_a_corner_handle_still_reshapes_in_box_mode(self):
        p = self._panel()
        p.box_radio.setChecked(True)
        p._ask_name = lambda default, error="": "B"
        self._drag_box(p, (40, 30), (220, 160))
        self._drag_box(p, (220, 160), (260, 200))     # starts on the bottom-right handle
        self.assertEqual(len(p.model.zones), 1, "must move the corner, not start a second box")
        self.assertEqual(p.model.zones[0].polygon[2], (260, 200))

    def test_switching_shape_or_mode_drops_half_drawn_points(self):
        p = self._panel()                              # polygon mode
        self._click(p, (10, 10))
        p.box_radio.setChecked(True)
        self.assertEqual(p.model.pending, [])

    def test_tripwire_finishes_automatically_at_two_points(self):
        p = self._panel()
        p.wire_radio.setChecked(True)
        p._ask_name = lambda default, error="": default
        self._click(p, (0, 120))
        self._click(p, (300, 120))
        self.assertEqual(len(p.model.tripwires), 1)
        self.assertEqual(p.model.pending, [])

    def test_clicks_in_the_letterbox_margin_are_ignored(self):
        p = self._panel()
        QTest.mouseClick(p.canvas, Qt.MouseButton.LeftButton, pos=QPoint(2, 2))
        x, y, w, h = p.canvas.transform().image_rect()
        margin_x = int(x / 2) if x > 4 else None
        if margin_x is not None:
            QTest.mouseClick(p.canvas, Qt.MouseButton.LeftButton, pos=QPoint(margin_x, int(y + h / 2)))
        self.assertEqual(p.model.pending, [])

    def test_duplicate_name_reprompts_with_the_error(self):
        p = self._panel()
        for pt in [(1, 1), (50, 1), (50, 50)]:
            p.model.add_point(pt)
        p.model.commit("Same")
        for pt in [(1, 1), (60, 1), (60, 60)]:
            p.model.add_point(pt)
        answers = iter(["Same", "Other"])
        errors = []
        def ask(default, error=""):
            errors.append(error)
            return next(answers)
        p._ask_name = ask
        p._name_and_commit()
        self.assertEqual([z.name for z in p.model.zones], ["Same", "Other"])
        self.assertIn("Duplicate", errors[1])

    def test_right_click_undoes_and_delete_selected_removes_a_shape(self):
        p = self._panel()
        self._click(p, (10, 10))
        QTest.mouseClick(p.canvas, Qt.MouseButton.RightButton, pos=QPoint(300, 300))
        self.assertEqual(p.model.pending, [])
        for pt in [(1, 1), (50, 1), (50, 50)]:
            p.model.add_point(pt)
        p.model.commit("Gone")
        p._refresh()
        p.shape_list.setCurrentRow(0)
        p._delete_selected()
        self.assertEqual(p.model.zones, [])

    def test_close_with_unsaved_changes_asks_first(self):
        p = self._panel()
        for pt in [(1, 1), (50, 1), (50, 50)]:
            p.model.add_point(pt)
        p.model.commit("X")
        closed = []
        p.closed.connect(lambda: closed.append(1))
        p._confirm_discard = lambda: False
        self.assertFalse(p.request_close())
        self.assertEqual(closed, [])
        p._confirm_discard = lambda: True
        self.assertTrue(p.request_close())
        self.assertEqual(closed, [1])

    def _add_zone(self, p, name="Z"):
        for pt in [(20, 20), (200, 30), (180, 200)]:
            p.model.add_point(pt)
        self.assertIsNone(p.model.commit(name))
        p.model.dirty = False
        p.canvas.update()

    def _w(self, p, img_pt):
        return QPoint(*(int(round(v)) for v in p.canvas.transform().to_widget(*img_pt)))

    def test_dragging_a_vertex_reshapes_the_zone_and_does_not_add_points(self):
        p = self._panel()
        self._add_zone(p)
        QTest.mousePress(p.canvas, Qt.MouseButton.LeftButton, pos=self._w(p, (200, 30)))
        QTest.mouseMove(p.canvas, self._w(p, (250, 60)))
        QTest.mouseRelease(p.canvas, Qt.MouseButton.LeftButton, pos=self._w(p, (250, 60)))
        self.assertEqual(p.model.zones[0].polygon[1], (250, 60))
        self.assertEqual(p.model.pending, [])
        self.assertTrue(p.dirty)
        self.assertIsNone(p.canvas._drag, "drag must end on release")

    def test_drag_is_clamped_to_the_frame(self):
        p = self._panel()
        self._add_zone(p)
        QTest.mousePress(p.canvas, Qt.MouseButton.LeftButton, pos=self._w(p, (200, 30)))
        QTest.mouseMove(p.canvas, QPoint(5000, 5000))
        QTest.mouseRelease(p.canvas, Qt.MouseButton.LeftButton, pos=QPoint(5000, 5000))
        self.assertEqual(p.model.zones[0].polygon[1], (319, 239))

    def test_shift_click_on_a_handle_starts_a_new_shape_instead(self):
        p = self._panel()
        self._add_zone(p)
        QTest.mouseClick(p.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier,
                         pos=self._w(p, (200, 30)))
        self.assertEqual(p.model.pending, [(200, 30)])

    def test_clicks_near_a_handle_while_drawing_add_points_not_drag(self):
        p = self._panel()
        self._add_zone(p)
        self._click(p, (300, 200))            # start a new shape away from handles
        self._click(p, (200, 30))              # lands on an existing handle mid-draw
        self.assertEqual(p.model.pending, [(300, 200), (200, 30)])

    def test_flip_button_reverses_the_selected_tripwire_and_canvas_paints_the_arrow(self):
        p = self._panel()
        p.wire_radio.setChecked(True)
        p._ask_name = lambda default, error="": "Gate"
        self._click(p, (0, 120))
        self._click(p, (300, 120))
        p._refresh()
        self.assertFalse(p.flip_btn.isEnabled(), "nothing selected yet")
        p.shape_list.setCurrentRow(0)
        self.assertTrue(p.flip_btn.isEnabled())
        before = (p.model.tripwires[0].p1, p.model.tripwires[0].p2)
        p.flip_btn.click()
        self.assertEqual((p.model.tripwires[0].p2, p.model.tripwires[0].p1), before)
        self.assertGreater(p.canvas.grab().width(), 0)   # paint (incl. arrow head) runs without error

    def test_entry_checkbox_makes_an_instant_alert_zone_and_greys_out_severity(self):
        p = self._panel()
        p.box_radio.setChecked(True)
        self.assertTrue(p.severity_combo.isEnabled())
        p.entry_check.setChecked(True)
        self.assertFalse(p.severity_combo.isEnabled(), "severity only sets a time limit, which entry zones don't use")
        p._ask_name = lambda default, error="": "Window"
        self._drag_box(p, (40, 30), (220, 160))
        self.assertTrue(p.model.zones[0].alert_on_entry)
        self.assertIn("entry alert", p.shape_list.item(0).text())
        self.assertGreater(p.canvas.grab().width(), 0)          # painting the entry colour/label works
        p.entry_check.setChecked(False)
        self.assertTrue(p.severity_combo.isEnabled())
        p._ask_name = lambda default, error="": "Yard"
        self._drag_box(p, (30, 170), (120, 230))
        self.assertFalse(p.model.zones[-1].alert_on_entry)

    def test_existing_zones_are_loaded_and_rescaled_to_the_new_frame(self):
        from src.config import Zone
        from src.zone_editor import ZoneEditorModel
        m = ZoneEditorModel([Zone("Old", [(100, 100), (200, 100), (200, 200)])], [], (640, 480))
        m.save(self.zones_path)
        p = self._panel()   # 320x240 frame
        self.assertEqual(p.model.zones[0].polygon[0], (50, 50))


class TestMainWindowEditFlow(ZoneEditingTestBase):
    def _window(self):
        from desktop_app.main_window import MainWindow
        w = MainWindow()
        w.zone_panel.path = self.zones_path
        w.file_path_edit.setText(self.clip)

        def join_threads():  # a QThread must not be garbage-collected while running
            for t in (w._snapshot_worker, w.worker):
                if t is not None:
                    t.wait(5000)
        self.addCleanup(join_threads)
        return w

    def test_edit_when_idle_uses_a_snapshot_then_saves_for_next_run(self):
        w = self._window()
        w._begin_edit_zones()
        self.assertFalse(w.edit_zones_btn.isEnabled(), "button locks while the frame is being fetched")
        self.assertTrue(_wait(lambda: w._editing), "snapshot never arrived")
        self.assertIs(w.preview_stack.currentWidget(), w.zone_panel)
        self.assertFalse(w.start_btn.isEnabled(), "Start must be off while editing")
        for pt in [(20, 20), (200, 30), (180, 200)]:
            w.zone_panel.model.add_point(pt)
        w.zone_panel._ask_name = lambda d, e="": "Lobby"
        w.zone_panel._name_and_commit()
        w.zone_panel.save()
        self.assertFalse(w._editing)
        self.assertTrue(w.start_btn.isEnabled())
        self.assertIs(w.preview_stack.currentWidget(), w.preview_label)
        self.assertIn("next run", w.status_bar.currentMessage())
        from src.config import load_zone_config
        self.assertEqual(load_zone_config(self.zones_path).zones[0].name, "Lobby")

    def test_edit_while_running_hot_applies_zones(self):
        _HookLayer.delay = 0.03
        w = self._window()
        statuses = []
        w._start()
        w.worker.status_changed.connect(statuses.append)
        self.assertTrue(_wait(lambda: w.worker.isRunning()))
        w._begin_edit_zones()
        self.assertTrue(_wait(lambda: w._editing), "running feed never handed over a frame")
        self.assertEqual(w.zone_panel.canvas.frame_size, (320, 240))
        for pt in [(20, 20), (200, 30), (180, 200)]:
            w.zone_panel.model.add_point(pt)
        w.zone_panel._ask_name = lambda d, e="": "Live"
        w.zone_panel._name_and_commit()
        w.zone_panel.save()
        self.assertIn("applied to the running feed", w.status_bar.currentMessage())
        self.assertTrue(_wait(lambda: any("Zones updated" in s for s in statuses)), statuses)
        w._stop()
        self.assertTrue(_wait(lambda: not w.worker.isRunning(), 10))

    def test_snapshot_failure_unlocks_the_button_and_does_not_enter_edit_mode(self):
        from PySide6.QtWidgets import QMessageBox
        shown = []
        orig = QMessageBox.warning
        QMessageBox.warning = staticmethod(lambda *a, **k: shown.append(a[2]))
        self.addCleanup(setattr, QMessageBox, "warning", orig)
        w = self._window()
        w.file_path_edit.setText(os.path.join(tempfile.gettempdir(), "_zone_edit_clip.mp4"))
        w._on_snapshot_failed("boom")
        self.assertFalse(w._editing)
        self.assertTrue(w.edit_zones_btn.isEnabled())
        self.assertEqual(shown, ["boom"])

    def test_unsolicited_snapshot_is_ignored(self):
        import numpy as np
        w = self._window()
        w._on_snapshot(np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertFalse(w._editing)


if __name__ == "__main__":
    unittest.main()
