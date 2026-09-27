# Reinforcement Learning Fundamentals

A reference for the concepts behind Automata — what reinforcement learning is, how DQN works, and how this harness turns those ideas into trained robot policies.

---

### What is Reinforcement Learning?

An agent interacts with an environment in a loop: it observes a **state**, takes an **action**, and receives a **reward**. The goal is to learn a **policy** — a mapping from states to actions — that maximizes cumulative reward over time.

```
          action
  Agent ---------> Environment
    ^                   |
    |   state, reward   |
    +-------------------+
```

Unlike supervised learning, where correct answers are provided, the agent discovers what works through trial and error. It must balance **exploration** — trying new things to discover better strategies — with **exploitation**, repeating what has worked so far.

---

### What is a DQN?

A Deep Q-Network is a neural network that learns to estimate the **Q-value** of each possible action in a given state. The Q-value represents the expected total future reward from taking that action and following the best policy thereafter.

```
State vector  -->  Neural network  -->  Q-values (one per action)
  [8 floats]        [2 hidden layers]     [Q₀, Q₁, Q₂, Q₃]
```

The agent picks the action with the highest Q-value. During training, it updates the network using the **Bellman equation**:

```
Q(state, action) = reward + gamma * max(Q(next_state, all_actions))
```

Here `gamma` is the discount factor, typically 0.99: it controls how much the agent values future rewards versus immediate ones.

The key insight: the network doesn't need to understand physics or strategy. Given enough experience, it learns which states lead to high rewards and which actions to take in those states.

---

### Key DQN Components

#### Replay Buffer

The agent stores every experience `(state, action, reward, next_state, done)` in a fixed-size buffer. During training, it samples random batches from this buffer rather than learning from experiences in order.

Why: consecutive experiences are correlated — step 100 looks a lot like step 101. Training on correlated data causes the network to overfit to recent experience and forget earlier lessons. Random sampling breaks this correlation.

In Automata: `core/memory.py` implements a circular buffer with configurable capacity, 10,000 transitions by default.

#### Epsilon-Greedy Exploration

The agent takes a random action with probability `epsilon`, and the greedy, highest-Q-value action otherwise. Epsilon starts high at 0.9 and decays toward a minimum of 0.01 over training.

```
Early training (epsilon = 0.9):  90% random, 10% greedy  →  exploring
Late training  (epsilon = 0.01):  1% random, 99% greedy  →  exploiting
```

This ensures the agent tries diverse strategies early on, then gradually commits to the best one it has found.

In Automata: `core/agent.py` handles action selection with `epsilon * decay` applied each episode.

#### Target Network

A second copy of the neural network, updated slowly by a soft update with `tau = 0.005`. The target network provides stable Q-value targets for the Bellman update. Without it, the network chases a moving target — every update changes both the prediction and the target, causing oscillation.

The soft update blends the online and target networks each step:

```
target_weights = (1 - tau) * target_weights + tau * online_weights
```

In Automata: `core/agent.py` maintains the target network and applies soft updates after every training step.

---

### What Does the Harness Do?

The harness is the infrastructure that connects these components into a training pipeline. It handles:

- **The training loop**, `core/train.py`: runs episodes, collects experience, calls the agent to learn
- **The agent**, `core/agent.py`: epsilon-greedy selection, replay sampling, network updates
- **The replay buffer**, `core/memory.py`: stores and samples experience
- **Checkpointing**: saves model weights periodically and tracks the best-performing model
- **Visualization**, `core/visualizer.py`: real-time reward chart during training

The harness is environment-agnostic — it only interacts with the environment through a standard interface of `reset()`, `step()` and `get_state()`. This means you can swap in any robot or task without modifying the training code.

---

### The Training Loop

```
for each episode:
    state = env.reset()
    while not done:
        action = agent.select_action(state, epsilon)
        next_state, reward, done, truncated, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.train()          # sample batch from replay buffer, Bellman update
        state = next_state
    decay epsilon
    checkpoint if best reward
    stop if solved
```

An episode is one complete run — from initial state to termination by crash, success, or time limit. The agent typically needs hundreds of episodes to converge on a good policy.

**Solve condition**: training stops when the average reward over the last `solve_window` episodes, 50 by default, exceeds `solve_threshold`. This is configurable per environment.

---

### Defining a Robot with MuJoCo

MuJoCo is a physics engine that simulates rigid-body dynamics — gravity, contacts, joints, actuators. Robots are defined in MJCF XML files that describe their physical structure:

```xml
<body name="quadrotor" pos="0 0 1">
  <freejoint/>                          <!-- 6-DOF: can move and rotate freely -->
  <inertial mass="0.5" .../>            <!-- mass and inertia -->
  <geom type="box" size="0.04 ..."/>    <!-- collision/visual geometry -->
  <site name="thrust_front" pos="0.15 0 0"/>  <!-- force application point -->
</body>

<actuator>
  <general site="thrust_front" gear="0 0 1 0 0 0"/>  <!-- upward thrust -->
</actuator>
```

MuJoCo handles all the physics — gravity, contact, rotation — and the 3D rendering. The environment subclass only needs to:

1. Map discrete actions to actuator commands
2. Extract observations from the simulation state
3. Compute a reward
4. Decide when to terminate

---

### Reward Function Design

The reward function is how you communicate the task objective to the agent. It's defined inside the environment's `step()` method.

A general pattern:

```python
reward = w1 * progress_toward_goal - w2 * unsafe_behavior
reward = clip(reward, -1, 1)
```

Design principles:

- **Reward the actual goal, not a proxy**: reward reaching the target, not just moving toward it
- **Balance components**: no single reward term should dominate the signal
- **Think adversarially**: ask "how would an optimizer game this?" — if you reward forward velocity, the agent may learn to run in circles
- **Clip to [-1, 1]**: large reward magnitudes destabilize training by causing large gradient updates
- **Only reward progress when safe**: e.g., only reward forward velocity when the robot isn't about to crash

---

### State Representation

The state is what the agent observes at each timestep. For robotics, this is typically a numerical vector extracted from simulator state or sensor readings:

| Source              | Example dimensions                                        | Use case                 |
| ------------------- | --------------------------------------------------------- | ------------------------ |
| Position + velocity | 6D: x, y, z, vx, vy, vz                                   | Basic locomotion         |
| + orientation       | 9-12D: add euler angles or a quaternion, plus angular velocity | Flight, balancing   |
| + sensor readings   | Variable: add contact, proximity, IMU                     | Manipulation, navigation |
| Camera image        | `(H, W, 3)` tensor                                        | Vision-based tasks       |

For vector states, use the MLP model in `models/mlp.py`. For image states, use the CNN in `models/cnn.py`. Both output raw Q-values — never softmax.

---

### Sim-to-Real

A policy trained in simulation will not transfer perfectly to real hardware. The gap comes from:

- **Physics mismatch**: simulated friction, mass, and damping don't match reality exactly
- **Sensor noise**: real sensors are noisy; simulated sensors are perfect
- **Latency**: real actuators have delays; simulated ones respond instantly

Strategies to close the gap, planned for future phases:

- **Domain randomization**: vary physics parameters — mass, friction, damping — randomly during training so the policy learns to be robust
- **Noise injection**: add Gaussian noise to observations during training to simulate sensor imperfection
- **System identification**: measure real hardware dynamics and tune the MJCF model to match
- **Fine-tuning on hardware**: run a few real-world episodes to adjust the policy

The MJCF file is the natural place to express these variations — the same robot description that defines the simulation also defines the randomization ranges.

---

### What the Trained Model Contains

When training saves `checkpoints/LunarLander-v3_best.pt`, it writes the neural network weights — the numbers that turn a state vector into Q-values:

```
{
  "net.0.weight": tensor (128, 8),    # 8 inputs -> 128 hidden
  "net.0.bias":   tensor (128,),
  "net.2.weight": tensor (128, 128),  # 128 -> 128 hidden
  "net.2.bias":   tensor (128,),
  "net.4.weight": tensor (4, 128),    # 128 -> 4 actions
  "net.4.bias":   tensor (4,),
}
```

This is the learned policy. To use it, reconstruct the same architecture from the config, load the weights, and pass states through it:

```python
action = model(state_tensor).argmax()  # pick the action with highest Q-value
```

The `.pt` file is Python/PyTorch-specific. A planned export to ONNX or TorchScript will make it portable to hardware that doesn't run Python.

---

### Beyond Vanilla DQN — What the Harness Adds

Four mechanisms ship in the harness for tasks where plain DQN stalls. Each is read from config
and each is a no-op at its default, so a simple environment is unaffected.

#### Double DQN — `training.double_dqn`, on by default

Vanilla DQN computes its target as `max` over the target network's own Q-values, so the same noisy
estimate both picks the next action and scores it. Noise that happens to favour an action is
therefore rewarded twice, which biases Q upward. The visible symptom is performance that rises,
peaks, and then decays. Double DQN lets the *online* network pick the next action and the *target*
network score it, which decouples selection from evaluation and removes the bias.

#### n-step returns — `training.n_step`, 1 by default

With 1-step bootstrapping, a reward earned at step 900 reaches the decision made at step 600 only
after credit has trickled back through 300 separate updates. n-step returns accumulate the next
`n` rewards into a single transition:

```
R = r0 + gamma*r1 + ... + gamma^(n-1)*r(n-1),  bootstrap gamma^n * Q(s_n)
```

Each transition therefore carries its own bootstrap discount, which is why the replay buffer
stores a discount per sample. Partial windows are flushed at the end of every episode so a return
never spans a reset.

#### Temporally-extended exploration — `training.explore_hold`, 1 by default

Per-step epsilon-greedy draws a fresh random action every step. In a task where the reward depends
on *where* the agent goes, independent random nudges cancel out: the agent explores jitter in
place, and never samples the sustained motion the task pays for. Holding one random action for
10–20 steps produces coherent exploratory manoeuvres instead. On travel-and-search tasks this is
usually the single largest lever — and it also raises the bar for "has it learned anything", since
a blind policy with held actions scores roughly twice a per-step random one.

#### Truncation is not termination

A timeout ends the episode for bookkeeping reasons; the world did not end. Its target must still
bootstrap `gamma*Q(s')`. Storing `done or truncated` as the terminal flag zeroes that bootstrap and
teaches the agent that long episodes are worth less than they are. The loop stores `done` only.

---

### Checkpoint Selection: Average Reward Is Not a Report Card

`checkpoints.best_metric` decides which checkpoint `_best.pt` keeps:

| Value | Tracks | Use when |
|---|---|---|
| `"reward"`, the default | rolling average reward | reward is the deliverable, as in most Gym tasks |
| `"success"` | rolling success rate from `info["is_success"]` | the task either completes or does not |
| any numeric `info` key | that key's rolling mean | the deliverable is a measured quantity such as area covered or time to goal |

The reason for the other two options: in any task where lingering in a good state is possible, a
policy that camps banks more return than one that does the job. Reward is the optimization target;
the scorecard should be the thing you actually want. The training loop never learns what the number
means — it maximizes whatever the environment reports, which is what keeps it environment-agnostic.
Selection on a non-reward metric waits until the rolling window is full, so an early lucky streak
cannot lock `_best.pt` to a nearly untrained network.

---

### Scoring a Policy Honestly

`eval.py` scores a checkpoint alongside three blind baselines on identical seeded episodes:

```bash
venv/bin/python eval.py --config configs/default.yaml --checkpoint checkpoints/X_best.pt \
    --policy checkpoint random hold straight --episodes 40 --seed 7000 --no-render
```

- **`random`** — uniform action each step.
- **`hold`** — random actions held for k steps. This, not `random`, is the honest floor.
- **`straight`** — one constant action per episode. A degenerate policy can score surprisingly well
  this way, which is why the baseline is permanent.
- **`Hact`** — action entropy computed within an episode then averaged; 0 means the policy plays one
  action and is ignoring its observations. Pooling across episodes hides exactly that collapse.

Select a checkpoint on one seed and confirm it on a disjoint one: scoring on the seeds you selected
on flatters the result. The full set of measurement rules is in `.claude/rules/evaluation.md`, and
the hand-coded ceiling that makes any of these numbers interpretable is in
`.claude/skills/solvability-ceiling`.

---

### Glossary

The vocabulary used in `CLAUDE.md` and `.claude/rules/`, in one place.

| Term | Meaning |
|---|---|
| **Q-value** | expected future reward for taking an action in a state |
| **Bellman update** | `Q(s,a) = reward + gamma * max(Q(s',a'))` |
| **Replay buffer** | stores past experiences, sampled randomly to break temporal correlation |
| **Epsilon-greedy** | random action with probability epsilon, greedy otherwise |
| **Held exploration** | keeping one random action for k steps, so exploration produces coherent motion instead of jitter in place |
| **n-step return** | folding the next n rewards into one transition, so a distant payoff propagates back n times faster |
| **Double DQN** | the online net picks the next action and the target net scores it, which removes the overestimation that makes performance peak then decay |
| **Blind floor** | the score a policy that ignores its observations already gets, measured by the `hold` baseline; the bar a learned policy must clear |
| **Action entropy**, `Hact` | how varied a policy's actions are *within* an episode; near 0 means it has collapsed to one action |
| **Hand-coded ceiling** | what a scripted controller with the same information achieves — the reference that separates a task problem from a learning problem |
| **Sim-to-real gap** | policies trained in simulation may not transfer cleanly to physical hardware; address it with domain randomization and noise injection |
