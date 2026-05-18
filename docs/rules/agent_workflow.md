# Agent Workflow Protocol

## 1. Purpose
This document defines the mandatory, step-by-step lifecycle that every AI agent or human contributor MUST follow when executing a task in the `RealTime-ViolenceDetection` project. This protocol ensures that decisions are properly tracked, logs are kept updated, and coding standards are rigidly applied.

## 2. The Execution Loop

Before beginning *any* coding, refactoring, or documentation task, you must follow this exact sequence:

### Step 1: Read the Constitution
*   **Action:** You MUST read `docs/ai_instructions.md` before doing anything else.
*   **Why:** It contains the non-negotiable rules, locked decisions, terminology disciplines, and checklists for safe code/doc modification.

### Step 2: Check Current State & Plan
*   **Action:** Read `docs/state.md` and `docs/project-plan.md`.
*   **Why:** `state.md` tells you what hyperparameters, shapes, and metrics are currently locked vs. TBD. `project-plan.md` tells you where we are in the development timeline.

### Step 3: Claim the Task
*   **Action:** Update the checklist in `docs/project-plan.md`.
*   **Rule:** Change the relevant task's status from `pending` to `in-progress`. If the task doesn't exist, add it under the appropriate phase.

### Step 4: Execute & Enforce Standards
*   **Action:** Write the code or documentation.
*   **Rule:** You MUST apply the principles defined in `docs/rules/coding_standards.md` (SOLID principles, Single Responsibility, zero magic numbers, proper docstrings).
*   **Rule:** Ensure you adhere strictly to the "Mandatory Preprocessing Chain" and other locked variables found in `ai_instructions.md`.

### Step 5: Update the Global State
*   **Action:** If you resolved a "TBD" item, discovered a new limitation, or added a new assumption during your execution:
*   **Rule:** You MUST update `docs/state.md` to reflect this new reality. Do not leave stale TBDs if you have implemented the solution.
*   **Rule:** If a locked decision had to be changed (which is rare and risky), you must update `decision_log.md` first, as instructed in `ai_instructions.md`.

### Step 6: Mark Complete & Log
*   **Action 1:** Update `docs/project-plan.md` to change the task status from `in-progress` to `done`.
*   **Action 2:** You MUST append a detailed execution log to `docs/project_log.md`.
*   **Rule:** Your log entry MUST strictly follow the markdown template provided in `docs/rules/logging_standarts.md`.

## 3. Workflow Summary Checklist for Agents

To avoid prompt-forgetfulness, agents should internally check off this list for every sub-task:
- [ ] Read `ai_instructions.md`.
- [ ] Checked `state.md` for locked constraints.
- [ ] Marked task `in-progress` in `project-plan.md`.
- [ ] Code complies with `coding_standards.md`.
- [ ] Updated `state.md` if any TBDs were resolved.
- [ ] Marked task `done` in `project-plan.md`.
- [ ] Appended formatted log to `project_log.md` per `logging_standarts.md`.

## 4. Failure to Comply
If an agent is found to have modified code without updating the project plan, leaving the state stale, or skipping the logging step, the execution is considered a **failure** and must be reverted and redone following this protocol.
