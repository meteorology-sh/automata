# Rule: Decide what to learn and what to write as code

The most expensive mistake available on an RL project is training a policy to do a job a hundred lines
of ordinary code already does better. Decide the split before the first run, and re-check it whenever a
hand-coded baseline beats the policy.

## Must follow

- **Split roles instead of asking one policy for two jobs.** A monolithic policy given two objectives
  hits a Pareto wall: every gain in one role is paid for out of the other, and no tuning crosses it.
  Splitting the roles dissolved that wall on this project and became the problem definition. If a policy
  is being asked to travel *and* to search, or to stabilize *and* to plan, separate them.

- **Give the learner only the part where a fixed pattern structurally cannot win.** Measured results,
  not opinion: an open-loop spiral matched or beat a trained DQN at acquiring a target in calm, moving
  and windy conditions, and a hand-coded lawnmower beat it at covering ground. Going to a known
  coordinate is waypoint navigation; sweeping a region is a geometric pattern. Neither needs learning.

- **When the target is large relative to your sensing precision, thorough coverage beats clever
  guessing.** A systematic sweep is *guaranteed* to pass through a target bigger than its leg spacing,
  while a reactive policy has to learn to be thorough. Check the ratio before assuming the problem is
  perception or policy quality.

- **Gate the layers on a physical signal, not on a learned decision.** "Is there a usable reading at
  all" is a threshold test. Hand over between a pattern and a policy on that, and keep each layer's code
  separate so either can be replaced or measured alone.

- **Measure each layer with its own metric, and never judge one layer by another's.** Whether the system
  reaches the region is the pattern layer's number; what the policy does once it has signal is the
  learned objective's number. Mixing them hides a win and manufactures a regression.

- **Bound the task by the platform's budget, not by ambition.** Compute what one episode physically buys
  — distance, time, energy — against what the task demands. When one sortie buys about 4.8 km of travel
  and the hardest setting needs 27 km, the failure is geometry, and training for it is wasted compute.
  Cap the curriculum where the hand-coded ceiling itself collapses.

- **Ground success tolerances and operating points in the deliverable.** Pick the number the job
  actually requires, then verify the sensor can resolve it and the platform can reach it. Do not pick
  the hardest number available and treat the gap as a learning problem.

- **"The classical controller wins" is a legitimate result — report it.** It is a finding about the task,
  not a failure of the work. Write it up per `.claude/rules/research-record.md`, keep the code, and point
  the learner at the part that is left.
