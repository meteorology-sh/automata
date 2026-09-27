---
paths:
  - "envs/*.py"
  - "envs/**/*.py"
---

# Rule: Modeling the physical platform

If the simulated vehicle is not honest about its own limits, every policy trained in it is optimizing a
fiction. Two things make it honest: real component data, and knowing which limit actually binds.

## Must follow

- **Keep the platform model in a pure module, and unit-test it.** Actuator curves, power draw, the
  environment's field or terrain model: plain functions over numbers, no simulator imports, no global
  state. Then the physics is testable without starting an episode, and a wrong number shows up as a
  failing test rather than as a policy that mysteriously will not fly. Read derived quantities such as
  mass from the loaded model rather than hardcoding them, so the env stays consistent with whatever
  description is loaded.

- **Use real component data, not a constant.** Take thrust, power and efficiency from the datasheet
  curve, and correct it for the operating conditions the task actually visits. A flat coefficient is
  usually wrong exactly where the task is hard.

- **Find out which constraint binds — measure it, do not reason about it.** We assumed a peak-capability
  ceiling would stop the vehicle and were wrong: the real limit was the consumable budget, and the
  correct design conclusion was the opposite of the intuitive one. Write a short script that computes
  the whole manoeuvre's cost under a few strategies and read the answer off it before designing around a
  limit that may not be the binding one.

- **Cost the whole manoeuvre, not the instantaneous rate.** A fast traversal can be strictly cheaper
  than a slow one once the standing idle cost is included, even though its instantaneous draw is far
  higher. Any argument about efficiency that looks only at the rate is likely to be backwards.

- **Respect duty limits.** Components have a sustainable operating band and a short-term band. Brief
  excursions are fine; a policy that lives at maximum output is proposing something the hardware will
  not do, so bound it in the model.

- **Apply disturbances as explicit external forces, separate from the actuators.** Drag, wind, contact
  pushes: compute them once per control step and add them as a body force. Keeping them out of the
  actuator path makes them individually testable and keeps the control authority honest.

- **Randomize per episode what you are unsure about.** Mass, drag coefficients, disturbance strength and
  heading drawn from `self.rng` after `_begin_episode()` are domain randomization for free, and they are
  the cheapest insurance against a policy that only works in one exact world. Keep the draw reproducible
  so an evaluation is still comparable.
