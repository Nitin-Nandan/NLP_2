---
trigger: always_on
---

---
name: global-execution-and-autonomy
description: Core operational directives for modularity, proactive skill generation, autonomous debugging, and user interruption thresholds.
---

# Rule: Global Execution & Autonomy

## 1. Modularity & Consistency
* **Architecture:** All code MUST be strictly object-oriented and separated into logical modules (e.g., `config.py`, `data_loader.py`, `model_utils.py`, `train.py`).
* **Anti-Pattern:** NEVER write monolithic scripts.
* **Coding Standards:** Maintain consistent variable naming (`snake_case`) and strict PEP-8 formatting across all phases.

## 2. Proactive Skill Creation
* **Trigger:** Preparing to execute a repetitive task or a complex standalone function (e.g., parsing logs, formatting tables, checking hardware).
* **Evaluation:** Assess if a reusable skill would be beneficial for future phases.
* **Action:** If beneficial, formulate a skill description and immediately invoke `skill-creator` to build it **BEFORE** proceeding with the main task.

## 3. Autonomous Debugging
* **Trigger:** A script or process fails.
* **Action:** IMMEDIATELY invoke the `systematic-debugging` skill.
* **Constraint:** Do NOT stop and wait for the user. You must attempt to fix the issue autonomously up to **3 times** before requesting user intervention.

## 4. Hard Stops (User Approval Requirements)
* **Directive:** You MUST halt execution and wait for the user's explicit approval **ONLY** under the following conditions:
  1. **Phase Completion:** A project phase is completely finished.
  2. **Architectural Shifts:** A core architectural change is required (e.g., re-initializing BERT embeddings).
  3. **High-Compute Tasks:** Initiating a training loop or executing a script that is expected to take longer than 2 minutes.
      * **Execution Protocol:** Do NOT execute the script yourself in the background. You must provide the exact terminal command to the user (e.g., `python src/train.py`), instruct the user to manually run it in their own terminal, and wait for the user to report back the final results before proceeding.

## 5. No Destructive Overwrites
* **Directive:** Never overwrite executable scripts from previous experiments or phases. 
* **Action:** When initiating a new experiment or phase, agents MUST create a new, distinct execution script (e.g., `train_[experiment].py`) or use parameterized CLI arguments to retain historical codebase states.