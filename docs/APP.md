# The app (`detkit app`)

```bash
.venv/bin/python -m detkit app --workdir runs/mine [--ckpt weights/grain_rfdetr_small_v1/final] [--port 7860] [--device cpu]
```
Open http://127.0.0.1:7860. Two tabs: **Review** and **Inference**.

The course is about **designing the data collection** so that the provided model can be used to estimate weight.
You are not expected to improve the detector by correcting labels and retraining it again and again. Use the
baseline model as given; correct boxes only where it serves your measurements (see below).

## Review tab
1. Pick a photo (list, Prev/Next, *Next unreviewed*).
2. Get boxes from the provided model (**Proposals** accordion):
   * *Propose with RF-DETR*: choose a checkpoint, score threshold, tile/overlap; *downscale* and *ROI* speed it up on CPU.
     Choose the photo's `crop` first.
3. Optional: edit boxes on the image: box tool + drag = new box; hand tool = move / resize / Delete; label tool changes the class.
   To delete a box: click its row in the box table (or type its `#`, e.g. `3` or `2,5`) and press **Delete box**,
   then **Save**.
4. Per-box `weight_g` and `box_notes` in the table under the image.
5. Fill the photo metadata form (`crop`, `pile_id`, `total_weight_g`, `manual_count`). Live stats show boxes per class and area.
   Grain moisture is assumed constant for all samples and is not recorded.
6. **Save** / **Save & next**. Saving marks the photo reviewed (it then enters the tables).

Correcting boxes is a measurement tool, not the goal: on a subset of photos, hand-corrected boxes give you a ground
truth to measure the detector's own error (`detkit eval --gt reviewed`, [EVAL.md](EVAL.md)), which is one of the error
sources in your weight estimate. A photo with no grains is valid: delete all boxes and save.
*Add a new crop / class* is in the form section.

Data is stored in `<workdir>/annotations/reviewed_coco.json`, `<workdir>/meta/image_meta.json`, photos in `<workdir>/photos`.

## Inference tab
Photo -> count -> weight -> cost.
1. Upload photos; optionally fill crop and `total_weight_g`.
2. Choose checkpoint, score threshold (use your best-F1 value), tile/overlap, ROI, downscale.
3. Choose the weight model. Constants and prices are editable under *Constants and prices* (saved in `weight_config.json`).
   Models you have not implemented yet show "student TODO".
4. Output: overlays, a table (counts, area, weight, interval, error vs measured, cost) and a CSV download.
5. *Save as samples* adds the photos to the project for review.

Calibrating the baseline constant is on the command line: `detkit weight calibrate` ([WEIGHT.md](WEIGHT.md)).

## After reviewing
Go to weight estimation ([WEIGHT.md](WEIGHT.md)). Retraining the detector (`detkit split --from-reviewed` ->
`detkit train`, [TRAINING.md](TRAINING.md)) is possible but optional and not part of the course.
