---
trigger: always_on
---

---
description: Ensures all execution happens within the 'nlp' conda environment on a Windows/PowerShell host.
globs: ["**/*.py", "environment.yml"]
---

# Environment & Windows Execution Standards

## Core Directives
1. **Windows Native Execution:** This project is hosted on **Windows**. You MUST use **PowerShell 7+** syntax for all terminal operations. 
   - Never use `ls`, `rm`, `cp`, or `export`. 
   - Use `Get-ChildItem`, `Remove-Item -Recurse -Force`, `Copy-Item`, and `$env:VAR = "val"`.
2. **The 'nlp' Mandate:** Every Python execution MUST be preceded by the Conda activation command for Windows.
   - **Correct Syntax:** `conda activate nlp; python path\to\script.py`
3. **Path Handling:** Always use backslashes `\` for shell commands, but ensure Python scripts handle paths using `pathlib` to maintain cross-compatibility.
4. **Dependency Failure Recovery:** - **Step A:** If a script fails with `ModuleNotFoundError`, verify the `nlp` env is active.
   - **Step B:** If missing, append the package to `environment` and run `conda env update --file environment.yml --prune` within the active `nlp` session.

## Output Constraint
If a command is corrected from Linux to PowerShell, the agent must include: **"Windows/PowerShell syntax enforced."**