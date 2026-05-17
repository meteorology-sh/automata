# PLAN — From Simulation to Hardware

## What Exists Today

The harness is a working DQN training and evaluation system:

- **Train** a DQN agent on any Gymnasium-compatible environment with a live dashboard (`python main.py`)
- **Evaluate** a trained agent with visual rendering (`python eval.py --config ... --checkpoint ...`)
- **Extend** with custom environments by subclassing `BaseEnv` and defining a reward function
- **Configure** everything from YAML — hyperparameters, model type, environment, action labels

Two environments are validated: CartPole-v1 and Acrobot-v1 (including a reward-shaped variant). See `CLAUDE.md` for architecture rules and extension guides.

---

## The Pipeline

The goal is a complete path from "I have a robot" to "it does the thing I trained it to do."

```
Define env          Train policy         Evaluate            Export model         Deploy
(simulator)    -->  (main.py)       -->  (eval.py)      -->  (.pt -> portable)  -->  (hardware)
    |                                                                                    |
    |                                                                                    |
    +------- Observe sim-to-real gap, tune simulator, retrain <--------------------------+
```

### Stage 1: Define the environment (supported)

Subclass `BaseEnv`. Map your robot's sensors to a state vector, its actuators to discrete actions, and define a reward function inside `step()`. For training, this environment talks to a *simulator* — not the real robot.

For robotics, the simulator is the critical piece. Options:
- **PyBullet** — free, good for arms and legged robots
- **MuJoCo** (via Gymnasium) — industry standard for contact-rich manipulation
- **Isaac Sim** — GPU-accelerated, massive parallelism, NVIDIA hardware

The simulator must produce the same state/action interface your real robot will. If the real robot has a 6-axis IMU and 4 motor channels, the simulator must output 6 floats and accept 4 discrete actions.

### Stage 2: Train a policy (supported)

```bash
python main.py --config configs/your_robot.yaml
```

The training loop runs episodes in the simulator, collecting experience and updating the neural network via DQN. The output is a `.pt` checkpoint file containing the trained weights.

### Stage 3: Evaluate in simulation (supported)

```bash
python eval.py --config configs/your_robot.yaml --checkpoint checkpoints/your_robot_best.pt
```

Watch the agent perform in the simulator with no exploration noise. This is where you verify the policy actually does what you want before putting it on hardware.

### Stage 4: Export the model (not yet built)

The `.pt` file is a Python-specific format — a dictionary mapping layer names (`net.0.weight`, `net.0.bias`, etc.) to PyTorch tensors. To run on hardware, it needs to be converted to something the target device can execute.

Options by target:
- **Linux SBC (Raspberry Pi, Jetson)** — TorchScript (`torch.jit.trace`) or ONNX. Both run without Python if needed.
- **Microcontroller (Arduino, STM32)** — Extract raw weight arrays and write a minimal forward pass in C. The MLP is just matrix multiplies and ReLU.
- **ROS2 node** — TorchScript or ONNX loaded in a C++ or Python node.

An `export.py` script would handle this: load a checkpoint + config, trace the model, save as TorchScript/ONNX.

### Stage 5: Deploy to hardware (not yet built)

The deployment loop is simpler than training — no replay buffer, no optimizer, no epsilon. It's a tight sensor-action cycle:

```
while running:
    state = read_sensors()         # IMU, cameras, encoders
    action = model.forward(state)  # neural network inference
    send_command(action)           # motor PWM, servo angles
    wait(control_period)           # e.g. 50ms for 20Hz control
```

A `deploy/infer.py` template would provide this loop with pluggable sensor and actuator interfaces.

### Stage 6: Close the sim-to-real gap (not yet built)

A policy trained in simulation will not work perfectly on real hardware. The simulator is always an approximation. Strategies:

- **Domain randomization** — vary physics parameters (friction, mass, latency) during training so the policy is robust to the real values
- **Noise injection** — add Gaussian noise to sensor readings during training to match real sensor noise
- **System identification** — measure real hardware dynamics and tune the simulator to match
- **Fine-tuning on hardware** — run a few real-world episodes to adjust the policy (requires safe exploration)

---

## Roadmap

### Phase 5 — Model export

> Goal: Convert trained checkpoints to portable formats for deployment.

- [ ] **5.1** Create `export.py` — load config + checkpoint, trace model with `torch.jit.trace`, save as TorchScript (`.ts`).
- [ ] **5.2** Add ONNX export path (`torch.onnx.export`) for non-PyTorch runtimes.
- [ ] **5.3** Add raw weight extraction for microcontroller targets — dump weights as C arrays or flat binary.
- [ ] **5.4** Test that exported models produce the same outputs as the original `.pt` for a set of test inputs.

### Phase 6 — Hardware inference template

> Goal: Provide a starting point for running a trained policy on real hardware.

- [ ] **6.1** Create `deploy/infer.py` — template inference loop with pluggable `read_sensors()` and `send_command()` functions.
- [ ] **6.2** Add a serial (UART) sensor/actuator backend as a reference implementation.
- [ ] **6.3** Add a ROS2 backend (publish actions to a topic, subscribe to sensor state).
- [ ] **6.4** Document control frequency considerations — the inference loop must run faster than the environment dynamics.

### Phase 7 — Domain randomization

> Goal: Make policies robust to the sim-to-real gap.

- [ ] **7.1** Add noise injection config to `BaseEnv` — Gaussian noise on state observations, configurable per-dimension.
- [ ] **7.2** Add physics randomization hooks — environments can vary parameters (mass, friction, latency) each episode.
- [ ] **7.3** Test that a domain-randomized CartPole policy is more robust to perturbation than a standard one.

### Phase 8 — Advanced algorithms

> Goal: Move beyond vanilla DQN for harder problems.

- [ ] **8.1** Target network (separate slowly-updated network for computing target Q-values — reduces training instability).
- [ ] **8.2** Double DQN (decouple action selection from value estimation to reduce overestimation bias).
- [ ] **8.3** Prioritized experience replay (sample important transitions more often).
- [ ] **8.4** Dueling DQN (separate state-value and advantage streams).

---

## What the .pt File Actually Is

When training saves `checkpoints/CartPole-v1_best.pt`, it writes a dictionary like:

```
{
  "net.0.weight": tensor of shape (128, 4),    # first layer: 4 inputs -> 128 hidden
  "net.0.bias":   tensor of shape (128,),
  "net.2.weight": tensor of shape (128, 128),  # second layer
  "net.2.bias":   tensor of shape (128,),
  "net.4.weight": tensor of shape (2, 128),    # output layer: 128 hidden -> 2 actions
  "net.4.bias":   tensor of shape (2,),
}
```

This is *what the agent learned* — the numeric weights that turn a state vector into Q-values. It is not a runnable program. To use it, you must:

1. Reconstruct the same model architecture (from the config)
2. Load the weights into it (`model.load_state_dict(...)`)
3. Pass a state through it (`model(state_tensor).argmax()` gives the best action)

This is what `eval.py` does. An export step (Phase 5) would convert this dictionary into a self-contained format that doesn't need Python or the config to run.
