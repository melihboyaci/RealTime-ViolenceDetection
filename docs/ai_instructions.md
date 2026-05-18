# ai_instructions.md

The most important file for any AI agent working on this project. Read it **before** touching code or docs.

## 1. Role and Mission

- You are a contributor to a **real-time, pose-based violence detection** system.
- Your mission is to advance the project **without violating** the locked decisions in `decision_log.md`.
- You serve **future agents** as much as the current human owner: every change must remain understandable to a coding agent that joins later.

## 2. Non-Negotiable Project Rules

1. The source PDF (summarized in `state.md`, `decision_log.md`, and across the docs) is the **only** source of truth for the locked baseline.
2. Do **not invent** numbers, metrics, hyperparameters, or experimental results.
3. If a value is missing, write **"TBD"** or **"Not specified in the source PDF."** — never guess.
4. Keep the **offline preprocessing** and **online inference** phases strictly separated.
5. Reuse the **same per-frame preprocessor and feature builder** in both phases. Drift between offline and online is a critical bug.
6. Never change a locked decision without updating `decision_log.md` and re-running the relevant ablation.

## 3. What to Read First (in this order)

1. `README.md` — project entry point.
2. `project.md` — vision and exec summary.
3. `state.md` — current state, locked decisions, open questions.
4. `decision_log.md` — authoritative table of locked decisions.
5. `architecture.md` — overall flow.
6. The doc most relevant to your task (`data_pipeline.md`, `model.md`, `training.md`, `inference.md`, `evaluation.md`, `limitations.md`, `specs.md`).

## 4. Safe vs. Unsafe Assumptions

| Safe to assume | Unsafe to assume |
|---|---|
| 10 FPS sampling target | Specific GRU hidden size or depth |
| 17 COCO keypoints from YOLOv8n-Pose | Specific learning rate / scheduler |
| 30-frame sequence windows | Specific early-stopping patience |
| 69-dim feature vector | Exact replacement value for low-confidence keypoints |
| Stratified 70 / 15 / 15 split | Exact stride of the sliding window |
| BCELoss + Adam, batch 32, ≤ 100 epochs | A specific motion statistic formula |
| Initial decision threshold 0.7 | A "tuned" deployment threshold |
| FIFO buffer length 30 | Real-time FPS / latency targets |
| Motion filter θ = 0.05, Violence only | Specific dataset folder layout |

When in doubt, treat the value as **TBD** and call it out explicitly.

## 5. How to Avoid Hallucinating Missing Details

- Search the docs for the term first; do not guess.
- If the term is not in any doc, mark it `TBD` in your output and ask the human (see §10).
- Do not fabricate references to the PDF.
- Do not invent file paths, module names, or class names that don't exist on disk.
- Do not invent metric numbers; use `TBD` placeholders.
- Do not extrapolate from related projects you "remember"; stay inside this project's docs.

## 6. How to Modify Docs Safely

- Update **one fact in one place**; if it appears elsewhere, propagate it identically.
- Always keep the **terminology and numbers** consistent with the table in `state.md`.
- When you change a locked decision: update `decision_log.md` first, then propagate.
- Never delete a section labeled "Assumptions / Open Questions"; update it instead.
- Prefer additive edits over rewrites when possible.

## 7. Terminology Discipline (Use Exactly These)

| Use | Do not use |
|---|---|
| Violence / NonViolence | violent / non-violent / fight / no-fight |
| offline preprocessing | data prep / pre-processing pipeline (alone) |
| online inference | real-time mode / streaming pipeline (alone) |
| feature vector | descriptor / embedding |
| sequence window | clip / chunk |
| motion filter | low-motion filter / activity filter |
| threshold | cutoff / decision boundary |

## 8. When to Ask for Human Clarification

Ask before acting if:

- A locked decision appears to conflict with new information.
- A value labeled `TBD` blocks meaningful progress.
- A change would touch more than one locked decision.
- The dataset on disk does not match the expected structure.
- The behavior of a fail-soft case (zero-person frame, sub-threshold keypoint) is not yet locked.

## 9. Checklist — Any Code Change

- [ ] The change does not violate any item in `decision_log.md`.
- [ ] The change uses the **same feature builder** offline and online (if applicable).
- [ ] Inputs and outputs match the shapes in `specs.md` (`(N, 69)` per video, `(30, 69)` per window).
- [ ] Failure modes from `specs.md` §7 (FH-1 … FH-6) are handled.
- [ ] No locked numbers (10 FPS, 0.5, 30, 69, 0.05, 0.7, 32, 100, 70/15/15) were silently changed.
- [ ] If a locked number changed, `decision_log.md` and the relevant ablation are updated.
- [ ] New TBD values are added to `state.md` Open Questions if they remain unresolved.
- [ ] The change is reproducible (seed, config, version pins).

## 10. Checklist — Any Documentation Change

- [ ] Terminology matches §7.
- [ ] Numbers match the table in `state.md`.
- [ ] Cross-file consistency: same fact appears identically in every file that mentions it.
- [ ] No invented metrics or hyperparameters.
- [ ] `Assumptions / Open Questions` section is up to date.
- [ ] Mermaid diagrams (if any) reflect the new flow.
- [ ] If a locked decision moved, `decision_log.md` is updated first.

## 11. Never Do

- ❌ Invent metric numbers, latency numbers, parameter counts, or hyperparameters.
- ❌ Drop a locked decision silently.
- ❌ Mix offline and online responsibilities (e.g. resampling FPS in the online path, or applying the motion filter at inference time).
- ❌ Use the **test split** for tuning anything, including the decision threshold.
- ❌ Apply the motion filter to **NonViolence** windows.
- ❌ Apply the motion filter at **inference time**.
- ❌ Save a checkpoint that is not the lowest-`val_loss` one and call it "the best".
- ❌ Use a different feature builder offline vs. online.
- ❌ Change keypoint confidence cutoff away from **0.5** without an ablation.
- ❌ Change the FIFO buffer length away from **30** without re-running training.

## 12. When in Doubt

- Pick the **safer, conservative** option (do nothing, or write a `TBD`).
- Prefer **explicitness over cleverness**.
- Prefer **fewer changes per PR**.
- Prefer **upstream fixes** over downstream workarounds.
- Prefer **asking the human** over guessing.
- If you must guess, label your assumption clearly inline and add it to `state.md` Open Questions.

## 13. Assumptions / Open Questions

- Whether agents are allowed to introduce new ablations beyond the four listed in `evaluation.md`: **TBD.**
- Whether agents may change the on-disk artifact layout independently: **TBD.**
- Whether agents may add a temporal smoothing layer to online inference: **TBD.**
