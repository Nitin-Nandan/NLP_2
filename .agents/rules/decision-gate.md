---
trigger: always_on
---

# Rule: User Decision Gate
## Description
Ensures the agent never executes long-running or architecturally significant changes without explicit user approval.

## Triggers
- Modifying neural network architectures (e.g., resizing embedding layers).
- Initiating a training loop expected to take longer than 2 minutes.
- Altering core hyperparameters (batch size, learning rate, epochs).

## Action
1. HALT execution immediately.
2. Present 2-3 distinct technical options to the user.
3. Detail the pros and cons of each option.
4. Wait for explicit user approval before writing code or running scripts.