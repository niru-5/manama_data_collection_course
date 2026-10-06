import json
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from detkit.app import logic as L
from detkit.app import proposals as P
from detkit.project import Project
from detkit.schema import IMAGE_FIELDS


@pytest.fixture
def wd(tmp_path):
    p = Project(workdir=tmp_path / "w", classes=["wheat", "sunflower"])
    p.save()
    p.photos_dir.mkdir()
    Image.new("RGB", (4000, 3000), (90, 90, 90)).save(p.photos_dir / "big.jpg")
    Image.new("RGB", (800, 600), (90, 90, 90)).save(p.photos_dir / "small.png")
    return p


def test_form_meta_coercion():
    vals = [None] * len(IMAGE_FIELDS)
    keys = [f.key for f in IMAGE_FIELDS]
    vals[keys.index("total_weight_g")] = "1.25"
    vals[keys.index("manual_count")] = 12.0
    vals[keys.index("pile_id")] = "  p1 "
    meta, errs = L.form_to_meta(vals)
    assert meta == {"total_weight_g": 1.25, "manual_count": 12, "pile_id": "p1"} and not errs
    vals[keys.index("total_weight_g")] = "abc"
    assert L.form_to_meta(vals)[1]
    # clearing a previously set field yields an explicit None
    vals = [None] * len(keys)
    assert L.form_to_meta(vals, {"pile_id": "old"})[0] == {"pile_id": None}
    assert L.meta_to_form({"pile_id": "x"})[keys.index("pile_id")] == "x"


def test_preview_and_scaling(wd):
    prev, size, scale = L.make_preview(wd.photos_dir / "big.jpg", wd.workdir / "c", 1000)
    assert size == (4000, 3000) and scale == 4.0 and Image.open(prev).size == (1000, 750)
    p2, s2, sc2 = L.make_preview(wd.photos_dir / "small.png", wd.workdir / "c", 1000)
    assert sc2 == 1.0 and p2.name == "small.png"


def test_box_roundtrip_and_edit_detection():
    classes = ["wheat", "sunflower"]
    boxes = [{"bbox": [400, 400, 800, 800], "class": "wheat", "origin": "rfdetr", "score": 0.9, "weight_g": 0.04},
             {"bbox": [1200, 1200, 1600, 1600], "class": "wheat", "origin": "vlm", "score": None}]
    ann = L.to_annotator(boxes, 4.0, classes)
    assert ann[0]["xmin"] == 100 and ann[0]["label"] == "wheat"
    same = L.from_annotator(ann, boxes, 4.0, (4000, 3000), classes)
    assert same == boxes                                     # untouched -> identical, extras kept
    ann[0]["xmax"] += 20                                     # resize
    ann[1]["label"] = "sunflower"                            # re-class
    ann.append({"xmin": 500, "ymin": 500, "xmax": 600, "ymax": 600, "label": "wheat"})   # add
    out = L.from_annotator(ann, boxes, 4.0, (4000, 3000), classes)
    assert out[0]["origin"] == "manual" and out[0]["weight_g"] == 0.04 and out[0]["bbox"][2] == 880
    assert out[1]["class"] == "sunflower" and out[1]["origin"] == "vlm"
    assert out[2]["origin"] == "manual" and out[2]["bbox"] == [2000, 2000, 2400, 2400]
    assert len(out) == 3
    del ann[0]                                               # delete
    assert len(L.from_annotator(ann, boxes, 4.0, (4000, 3000), classes)) == 2
    # clamping / degenerate boxes
    bad = [{"xmin": -5, "ymin": 0, "xmax": 2, "ymax": 0.1, "label": "wheat"}]
    assert L.from_annotator(bad, [], 1.0, (100, 100), classes) == []


def test_table_edits_and_stats():
    boxes = [{"bbox": [0, 0, 10, 10], "class": "wheat", "origin": "manual"},
             {"bbox": [0, 0, 20, 10], "class": "sunflower", "origin": "vlm", "score": 0.5}]
    rows = L.boxes_to_rows(boxes)
    rows[0][-2] = "0.04"
    rows[1][-2] = 0.06
    rows[1][-1] = "big"
    out, errs = L.apply_table_edits(boxes, rows)
    assert not errs and out[0]["weight_g"] == 0.04 and out[1]["box_notes"] == "big"
    rows[0][-2] = "x"
    assert L.apply_table_edits(boxes, rows)[1]
    st = L.compute_stats(out, {"scale_mm_per_px": 0.5, "total_weight_g": 0.1}, ["wheat", "sunflower"])
    assert st["n"] == 2 and st["total_area_px"] == 300 and st["total_area_mm2"] == 75
    assert st["sum_box_weight_g"] == pytest.approx(0.1) and "OK" in st["weight_hint"]
    assert "MORE" in L.weight_hint(2, 0.05, 2, 0.1) and "per grain" in L.weight_hint(4, 1.0, 0, None)
    assert "differs" in L.weight_hint(3, None, 0, None, manual_count=5)


def test_delete_boxes():
    boxes = [{"bbox": [0, 0, i + 1, i + 1], "class": "wheat"} for i in range(4)]
    left, gone = L.delete_boxes(boxes, "2, 4")
    assert gone == [2, 4] and [b["bbox"][2] for b in left] == [1, 3]
    assert L.delete_boxes(boxes, "1")[0] == boxes[1:]
    for bad in ("", "5", "0", "x"):
        with pytest.raises(ValueError):
            L.delete_boxes(boxes, bad)


def test_add_class_appends_and_persists(wd):
    L.add_class(wd, " corn ")
    q = Project.load(wd.workdir)
    assert q.classes == ["wheat", "sunflower", "corn"]
    for bad in ("", "Wheat", "a,b"):
        with pytest.raises(ValueError):
            L.add_class(wd, bad)


def test_import_photos_no_overwrite(wd, tmp_path):
    src = tmp_path / "small.png"
    Image.new("RGB", (10, 10)).save(src)
    (tmp_path / "x.txt").write_text("no")
    assert L.import_photos([src, tmp_path / "x.txt"], wd.photos_dir) == ["small_1.png"]
    assert L.list_photos(wd.photos_dir) == ["big.jpg", "small.png", "small_1.png"]


def test_combine_modes():
    ex = [{"origin": "manual", "bbox": [0, 0, 1, 1]}, {"origin": "vlm", "bbox": [1, 1, 2, 2]}]
    new = [{"origin": "rfdetr", "bbox": [3, 3, 4, 4]}]
    assert [b["origin"] for b in L.combine(ex, new, L.MODES[0])] == ["manual", "rfdetr"]
    assert len(L.combine(ex, new, "replace all")) == 1 and len(L.combine(ex, new, "append")) == 3


def test_label_mapping_and_dets():
    cl = ["wheat", "sunflower"]
    assert P.map_label("Wheat", cl, None, False) == "wheat"
    assert P.map_label("seed", cl, "sunflower", True) == "sunflower"
    assert P.map_label("seed", cl, "sunflower", False) is None
    dets = [{"bbox": [10, 20, 30, 40], "label": "seed", "score": 0.8}, {"bbox": [0, 0, 5, 5], "label": "??", "score": 0.5}]
    boxes, dropped = P.dets_to_boxes(dets, cl, "wheat", origin="rfdetr", single_class_model=False, offset=(100, 200), factor=0.5)
    assert boxes == [] and dropped == 2
    boxes, dropped = P.dets_to_boxes(dets, cl, "wheat", origin="rfdetr", single_class_model=True, offset=(100, 200), factor=0.5)
    assert boxes[0]["bbox"] == [120, 240, 160, 280] and boxes[0]["class"] == "wheat" and dropped == 0


def test_propose_rfdetr_fake_predictor():
    seen = {}

    def fake_predict(im, model, proc, device, *, tile, overlap, score):
        seen.update(size=im.size, tile=tile, score=score)
        return [{"bbox": [10, 10, 50, 50], "score": 0.9, "label": "seed", "label_id": 0}]

    model = SimpleNamespace(config=SimpleNamespace(id2label={0: "seed"}))
    bundle = (model, None, {}, "cpu")
    img = Image.new("RGB", (4000, 2000))
    boxes, info = P.propose_rfdetr(img, bundle, ["wheat", "sunflower"], "sunflower", score=0.4, tile=800,
                                   max_side=1000, roi=(1000, 500, 3000, 1500), predict_fn=fake_predict)
    assert seen["size"] == (1000, 500) and seen["score"] == 0.4          # ROI 2000x1000 -> factor 0.5
    assert boxes[0]["bbox"] == [1020, 520, 1100, 600] and boxes[0]["class"] == "sunflower" \
        and boxes[0]["origin"] == "rfdetr"
    with pytest.raises(ValueError):                                       # single-class model, unknown crop
        P.propose_rfdetr(img, bundle, ["wheat", "sunflower"], None, predict_fn=fake_predict)
    assert P.parse_roi("", (10, 10)) is None and P.parse_roi("0,0,99999,50", (100, 80)) == (0, 0, 100, 50)


def test_find_checkpoints(tmp_path):
    for n in ("checkpoints/r1/final", "weights/base/final"):
        d = tmp_path / n
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}")
    got = P.find_checkpoints(tmp_path)
    assert str(tmp_path / "checkpoints/r1/final") in got and str(tmp_path / "weights/base/final") in got


def test_save_review_store_roundtrip(wd):
    r = L.save_review(wd, "big.jpg", [
        {"bbox": [100, 100, 200, 220], "class": "wheat", "origin": "rfdetr", "score": 0.7, "weight_g": 0.05},
        {"bbox": [-50, -50, 0.2, 0.2], "class": "wheat"}],              # degenerate -> dropped
        {"crop": "wheat", "total_weight_g": "1.5"}, proposer="rfdetr")
    assert r["boxes"] == 1 and r["size"] == [4000, 3000]
    boxes, meta = wd.store().get_image("big.jpg")
    assert boxes[0]["bbox"] == [100, 100, 200, 220] and boxes[0]["weight_g"] == 0.05
    assert meta["reviewed"] and meta["total_weight_g"] == 1.5 and meta["proposer"] == "rfdetr"
    with pytest.raises(ValueError):
        L.save_review(wd, "big.jpg", [{"bbox": [1, 1, 5, 5], "class": "corn"}], {})
    rows = L.photo_rows(L.list_photos(wd.photos_dir), wd.store())
    assert rows[0][:3] == ["big.jpg", "reviewed", 1] and rows[1][1] == "todo"


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_app_starts_and_saves_via_client(wd):
    pytest.importorskip("gradio")
    client_mod = pytest.importorskip("gradio_client")
    from detkit.app.ui import build_app

    demo = build_app(wd.workdir)
    port = _free_port()
    demo.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True,
                allowed_paths=[str(wd.workdir)])
    try:
        c = client_mod.Client(f"http://127.0.0.1:{port}", verbose=False)
        stt = c.predict(api_name="/status")
        assert stt["photos"] == ["big.jpg", "small.png"] and stt["classes"] == ["wheat", "sunflower"]
        boxes = json.dumps([{"bbox": [10, 10, 90, 70], "class": "sunflower", "weight_g": 0.03}])
        r = c.predict("small.png", boxes, json.dumps({"crop": "sunflower", "total_weight_g": 0.5}), api_name="/save_review")
        assert r["boxes"] == 1
        assert c.predict(api_name="/status")["reviewed"] == ["small.png"]
        b, m = wd.store().get_image("small.png")
        assert b[0]["class"] == "sunflower" and b[0]["origin"] == "manual" and m["total_weight_g"] == 0.5
    finally:
        demo.close()


# ---- inference tab -----------------------------------------------------------------------------
from detkit import weight as W                      # noqa: E402
from detkit.app import inference as I               # noqa: E402


def _fake_propose(monkeypatch, n=4):
    """Replace checkpoint loading + RF-DETR with a deterministic fake (no torch, no GPU)."""
    monkeypatch.setattr(P, "load_bundle", lambda ckpt, device="auto": ("bundle", ckpt, device))

    def fake(image, bundle, classes, crop, **kw):
        cls = crop or classes[0]
        return ([{"bbox": [10 + 60 * i, 10, 50 + 60 * i, 60], "class": cls, "origin": "rfdetr", "score": 0.9}
                 for i in range(n)], {})

    monkeypatch.setattr(P, "propose_rfdetr", fake)


def test_default_score_and_ckpt_defaults(wd, tmp_path):
    assert I.default_score(wd.workdir)[0] == 0.5
    rep = wd.workdir / "reports" / "eval_x"
    rep.mkdir(parents=True)
    (rep / "eval.json").write_text(json.dumps({"best_f1": {"score": 0.1873}}))
    assert I.default_score(wd.workdir)[0] == 0.187
    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "detkit_meta.json").write_text(json.dumps({"tile_params": {"tile": 640, "overlap": 0.1}}))
    assert I.ckpt_defaults(str(ck), wd) == {"tile": 640, "overlap": 0.1}
    assert I.ckpt_defaults(None, wd)["tile"] == wd.tile


def test_estimate_row_baseline_todo_and_error():
    dets = [{"bbox": [0, 0, 10, 10], "score": 0.9, "label": "wheat"}] * 5
    cfg = W.load_config("/nonexistent")
    row = I.estimate_row("a.jpg", dets, {"crop": "wheat", "total_weight_g": 0.25}, W.CountTimesConstant(cfg["constants"]), cfg)
    assert row["count"] == 5 and row["weight_g"] == 0.2 and row["error_pct"] == -20.0
    assert row["cost"] == pytest.approx(0.2 / 1000 * cfg["price_per_kg"]["default"], abs=1e-4) and row["counts"] == "wheat 5"
    todo = I.estimate_row("a.jpg", dets, {"crop": "wheat"}, W.get_model("linear_area"), cfg)
    assert todo["weight_g"] is None and "docs/STUDENT_TASKS.md task 2" in todo["note"] and todo["count"] == 5
    assert any("student TODO" in label for label, _ in I.model_choices())


def test_results_csv():
    rows = [{"file": "a.jpg", "counts": "wheat 2", "count": 2, "weight_g": 0.08, "note": ""}]
    text = I.results_csv(rows)
    assert text.splitlines()[0].startswith("file,counts,count") and "a.jpg,wheat 2,2" in text.splitlines()[1]


def test_run_inference_and_save_samples_roundtrip(wd, tmp_path, monkeypatch):
    _fake_propose(monkeypatch, n=4)
    src = tmp_path / "up"
    src.mkdir()
    for n in ("small.png", "new.png"):
        Image.new("RGB", (800, 600), (80, 80, 80)).save(src / n)
    rows, items, ovs = I.run_inference(
        [src / "small.png", src / "new.png"], {"small.png": {"crop": "wheat", "total_weight_g": 0.2}},
        wd, ckpt="ck", device="cpu", score=0.3, tile=800, overlap=0.2, max_side=0, roi_text="",
        model_name="count_x_constant", cache=tmp_path / "cache")
    assert [r["count"] for r in rows] == [4, 4] and rows[0]["measured_g"] == 0.2 and rows[0]["weight_g"] == 0.16
    assert all(Path(o).exists() for o in ovs) and rows[1]["counts"] == "wheat 4"      # crop defaults to a class
    names = I.save_samples(wd, items)
    assert names == ["small_1.png", "new.png"]                                          # clash -> suffix, no overwrite
    boxes, meta = wd.store().get_image("small_1.png")
    assert len(boxes) == 4 and boxes[0]["origin"] == "rfdetr" and boxes[0]["score"] == 0.9
    assert meta["reviewed"] is False and meta["total_weight_g"] == 0.2
    assert meta["crop"] == "wheat" and (wd.photos_dir / "small.png").exists()


def test_save_constants(wd):
    assert "Saved" in I.save_constants(wd, [["sunflower", 0.07, 0.3], ["default", "", None]])
    assert W.load_config(wd.workdir)["constants"]["sunflower"] == 0.07
    assert I.constants_rows(wd)[0][0] == "default"


def test_infer_via_client(wd, monkeypatch, tmp_path):
    client_mod = pytest.importorskip("gradio_client")
    _fake_propose(monkeypatch, n=3)
    ck = tmp_path / "ck"
    ck.mkdir()
    from detkit.app.ui import build_app

    demo = build_app(wd.workdir)
    port = _free_port()
    demo.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True,
                allowed_paths=[str(wd.workdir)])
    try:
        c = client_mod.Client(f"http://127.0.0.1:{port}", verbose=False)
        r = c.predict(json.dumps(["small.png"]), str(ck), "count_x_constant", 0.3, "wheat", 0.1, True, api_name="/infer")
        assert r["rows"][0]["count"] == 3 and r["rows"][0]["weight_g"] == pytest.approx(0.12)
        assert r["saved"] == ["small_1.png"]
        assert wd.store().get_image("small_1.png")[1]["reviewed"] is False
    finally:
        demo.close()
