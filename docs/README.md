# Reinforcement Learning Harness — Key Ingredients

A general reference for building an agentic RL framework that can be adapted to any robotics system.

---

## The Core Idea

An RL harness connects three things: a **perception model** (how the agent sees), a **decision model** (how the agent acts), and an **environment** (what the agent interacts with). Given any robotics system, your job is to define each of these clearly before writing any code.

---

## Ingredient 1 — State Representation

The state is what the agent observes at each timestep. Ask:

- What sensors does the robot have? (camera, lidar, IMU, GPS)
- What format is the data in? (image tensor, numerical vector, point cloud)
- Does the agent need memory? (a single frame, or a sequence of frames)

**If the state is an image**, preprocess it into a normalized tensor:

1. Convert color space if needed (BGR → RGB for PyTorch)
2. Resize to a fixed resolution (224×224 for pretrained CNNs, smaller for speed)
3. Normalize pixel values to [0, 1]
4. Reshape to `(batch, channels, height, width)`

**If the state is a numerical vector** (e.g. position, velocity, sensor readings), normalize each dimension to a consistent range and feed directly to a fully connected network.

---

## Ingredient 2 — Perception Model (CNN)

Used when the state contains visual input. The CNN extracts meaningful features from raw pixels so the decision model doesn't have to reason about individual pixel values.

```
Input image → Convolutional layers → Feature maps → Flatten → Feature vector
```

Key decisions:

- **Pretrained vs from scratch**: use a pretrained backbone (ResNet, EfficientNet) when you need object recognition or have limited data. Train from scratch when the visual domain is very different from natural images (e.g. aerial maps, thermal imaging).
- **Output**: the CNN should output a flat feature vector, not class probabilities. No softmax.
- **Input resolution**: higher resolution preserves detail but increases compute. Match to task complexity.

---

## Ingredient 3 — Decision Model (DQN or Policy Network)

Takes the feature vector from the CNN (or raw state vector) and outputs action values.

**Deep Q-Network (DQN)** — outputs one Q-value per action. Agent picks the highest.

- Best for: discrete action spaces (turn left, turn right, stop)
- Output: raw logits, no softmax

**Policy gradient methods** (e.g. PPO) — outputs a probability distribution over actions.

- Best for: continuous action spaces (exact thrust, exact angle)
- Better for complex real-world robotics but harder to implement

For most robotics systems, start with DQN if actions are discrete, PPO if continuous.

---

## Ingredient 4 — Environment Interface

Wrap your simulator (or real robot) in a Gymnasium-compatible interface:

```python
class RobotEnv(gym.Env):
    def reset(self):
        # return initial state, info

    def step(self, action):
        # apply action, return (next_state, reward, done, truncated, info)

    def render(self):
        # optional visualization
```

Key decisions:

- **Action space**: what can the robot actually do? Define minimum viable actions.
- **Episode termination**: when does a run end? (crash, goal reached, time limit)
- **Simulator fidelity**: higher fidelity = better sim-to-real transfer, more compute cost

---

## Ingredient 5 — Reward Function

The only way you communicate intent to the agent. Design it carefully.

A general template:

```python
reward = 0
reward += w1 * progress_toward_goal     # what you want
reward -= w2 * unsafe_behavior          # what you don't want
reward += w3 * int(goal_reached)        # terminal success
reward -= w4 * int(failed)              # terminal failure
```

Principles:

- **Think adversarially**: ask "how would an amoral optimizer game this reward?"
- **Balance components**: no single term should dominate
- **Prefer goal rewards over proxy rewards**: reward reaching the destination, not just moving forward
- **Clip rewards** to a fixed range (e.g. [-1, 1]) for training stability

---

## Ingredient 6 — Training Loop

The standard DQN training loop:

```
for each episode:
    state = env.reset()
    while not done:
        action = select_action(state, epsilon)   # epsilon-greedy
        next_state, reward, done = env.step(action)
        memory.append((state, action, reward, next_state, done))
        train_on_batch(memory)                   # sample random batch, Bellman update
        state = next_state
    decay epsilon
```

Key components:

- **Replay buffer**: stores past experiences, breaks temporal correlation, enables experience reuse
- **Epsilon-greedy**: start ~1.0 (random), decay toward ~0.01 (greedy). Controls explore/exploit balance
- **Batch training**: sample randomly from replay buffer each step, not sequentially
- **Target network**: a slowly updated copy of the model used to compute `target_q`, prevents instability (omitted in basic implementations, important in production)

---

## Ingredient 7 — Stopping Criteria and Checkpointing

Don't rely on loss alone — use episode reward as the primary signal.

```python
if avg_reward_last_100_episodes > threshold:
    # solved

torch.save(model.state_dict(), "checkpoint.pt")  # save periodically
```

Stopping criteria:

1. Performance threshold reached
2. Reward has plateaued
3. Compute budget exhausted

Always save the best-performing checkpoint, not just the final one.
