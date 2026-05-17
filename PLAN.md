# PLAN — From Simulation to Hardware

## What Exists Today

The harness is a working DQN training and evaluation system:

- **Train** a DQN agent on any Gymnasium-compatible environment with a live dashboard (`python main.py`)
- **Evaluate** a trained agent with visual rendering (`python eval.py --config ... --checkpoint ...`)
- **Extend** with custom environments by subclassing `BaseEnv` and defining a reward function
- **Configure** everything from YAML — hyperparameters, model type, environment, action labels

Two environments are validated: CartPole-v1 and Acrobot-v1 (including a reward-shaped variant). A custom-physics drone hover environment exists as a proof-of-concept (`envs/drone_hover.py`) but uses matplotlib rendering instead of a proper simulator — this is the gap that MuJoCo integration fills.

---

## Key Finding: MuJoCo as the Physics Backend

Early work with drone environments revealed that **PyBullet is effectively unmaintained** (last release 2022, no pre-built wheels for Python 3.12, fails to build on Windows). PyFlyt and other PyBullet-based robotics environments inherit this problem.

**MuJoCo** is the modern replacement. DeepMind open-sourced it, actively maintains it, and publishes pre-built wheels for all platforms. It is the standard physics backend for robotics RL research (manipulation, locomotion, flight). Gymnasium's robotics suite is built on it.

This means the harness should adopt MuJoCo as its first-class simulator, with robot definitions provided as MJCF XML files.

---

## The Three Inputs

A trained policy is produced from three inputs that compose cleanly:

```
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│   MJCF XML       │   │  BaseEnv subclass │   │   Config YAML    │
│                  │   │                  │   │                  │
│  Robot geometry  │   │  Reward function │   │  Hyperparameters │
│  Joints/bodies   │   │  Observation map │   │  Model arch      │
│  Actuators       │   │  Action mapping  │   │  Epsilon/gamma   │
│  Sensors         │   │  Termination     │   │  Solve threshold │
└────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
         │                      │                       │
         └──────────────────────┼───────────────────────┘
                                │
                         ┌──────▼──────┐
                         │  main.py    │
                         │  (train)    │
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │  .pt model  │
                         └─────────────┘
```

- **MJCF XML** — defines *what* the robot is. Geometry, mass, joints, actuators, sensors, contact properties. MuJoCo handles rendering and physics from this file alone.
- **BaseEnv subclass** — defines *what the robot should learn*. Maps simulator state to observations, discrete actions to actuator commands, and computes the reward. This is where the task lives.
- **Config YAML** — defines *how to train*. Hyperparameters, model architecture, epsilon schedule, solve criteria. Environment-agnostic.

Adding a new robot means dropping in an MJCF file and writing a thin env subclass. No changes to the training loop, model, or agent.

---

## The Pipeline

```
Define robot        Define task         Train policy         Evaluate            Export           Deploy
(MJCF XML)    -->  (BaseEnv)      -->  (main.py)       -->  (eval.py)      -->  (.pt → ONNX)  -->  (hardware)
    |                                                                                                  |
    |                                                                                                  |
    +------- Observe sim-to-real gap, tune MJCF + domain randomization, retrain <---------------------+
```

### Stage 1: Define the robot (new — MuJoCo integration)

Write an MJCF XML file describing the robot. MuJoCo's XML format defines bodies, joints, actuators, sensors, and visual geometry. For a quadrotor this would include the frame body, 4 rotor actuators, an IMU sensor, and optional camera sensors.

MuJoCo provides:
- **Physics** — contact dynamics, constraints, aerodynamics
- **Rendering** — 3D visualization via `render_mode="human"`, no custom code needed
- **Deterministic simulation** — reproducible rollouts for debugging

The MJCF file path is specified in the config YAML under `env.mjcf`.

### Stage 2: Define the task (supported)

Subclass `BaseEnv`. The subclass wraps the MuJoCo environment and defines:
- How to construct observations from simulator state (which sensors, what normalization)
- How to map discrete DQN actions to continuous actuator commands
- The reward function inside `step()`
- Termination conditions (crash, out of bounds, goal reached)

The same MJCF robot can support multiple tasks (hover, waypoint tracking, landing) by swapping the env subclass.

### Stage 3: Train a policy (supported)

```bash
python main.py --config configs/your_robot.yaml
```

The training loop runs episodes in the simulator, collecting experience and updating the neural network via DQN. The output is a `.pt` checkpoint file containing the trained weights.

### Stage 4: Evaluate in simulation (supported)

```bash
python eval.py --config configs/your_robot.yaml --checkpoint checkpoints/your_robot_best.pt
```

Watch the agent perform in the simulator with no exploration noise. MuJoCo renders the robot in 3D automatically when `render_mode="human"` is set. This is where you verify the policy before putting it on hardware.

### Stage 5: Export the model (not yet built)

The `.pt` file is a Python-specific format. To run on hardware, convert to a portable format:

- **Linux SBC (Raspberry Pi, Jetson)** — TorchScript (`torch.jit.trace`) or ONNX
- **Microcontroller (Arduino, STM32)** — extract raw weight arrays, write a minimal forward pass in C
- **ROS2 node** — TorchScript or ONNX loaded in a C++ or Python node

### Stage 6: Deploy to hardware (not yet built)

The deployment loop is simpler than training — no replay buffer, no optimizer, no epsilon:

```
while running:
    state = read_sensors()         # IMU, cameras, encoders
    action = model.forward(state)  # neural network inference
    send_command(action)           # motor PWM, servo angles
    wait(control_period)           # e.g. 50ms for 20Hz control
```

### Stage 7: Close the sim-to-real gap (not yet built)

A policy trained in simulation will not work perfectly on real hardware. Strategies:

- **Domain randomization** — vary physics parameters (friction, mass, latency) in the MJCF during training
- **Noise injection** — add Gaussian noise to sensor readings during training
- **System identification** — measure real hardware dynamics and tune the MJCF to match
- **Fine-tuning on hardware** — run a few real-world episodes to adjust the policy

The MJCF file is the natural place to express domain randomization — vary mass, inertia, damping, and actuator gain ranges per episode.

---

## Roadmap

### Phase 5 — MuJoCo integration

> Goal: Make MuJoCo the first-class physics and rendering backend.

- [ ] **5.1** Add `mujoco` to requirements. Verify `pip install mujoco` works on the target platform.
- [ ] **5.2** Create `envs/mujoco_env.py` — a `BaseEnv` subclass that wraps `gymnasium.make()` with a MuJoCo environment, handling render mode passthrough and the continuous-to-discrete action mapping pattern.
- [ ] **5.3** Add MJCF config support — `env.mjcf` key in YAML points to the XML file, passed to `gym.make()`.
- [ ] **5.4** Build a quadrotor MJCF model (frame, 4 rotors, IMU) and a `DroneHover` env subclass that uses it. Replaces the current custom-physics drone.
- [ ] **5.5** Validate: train drone hover with MuJoCo rendering, confirm eval renders the 3D drone.

### Phase 6 — Model export

> Goal: Convert trained checkpoints to portable formats for deployment.

- [ ] **6.1** Create `export.py` — load config + checkpoint, trace model with `torch.jit.trace`, save as TorchScript (`.ts`).
- [ ] **6.2** Add ONNX export path (`torch.onnx.export`) for non-PyTorch runtimes.
- [ ] **6.3** Add raw weight extraction for microcontroller targets — dump weights as C arrays or flat binary.
- [ ] **6.4** Test that exported models produce the same outputs as the original `.pt` for a set of test inputs.

### Phase 7 — Hardware inference template

> Goal: Provide a starting point for running a trained policy on real hardware.

- [ ] **7.1** Create `deploy/infer.py` — template inference loop with pluggable `read_sensors()` and `send_command()` functions.
- [ ] **7.2** Add a serial (UART) sensor/actuator backend as a reference implementation.
- [ ] **7.3** Add a ROS2 backend (publish actions to a topic, subscribe to sensor state).
- [ ] **7.4** Document control frequency considerations — the inference loop must run faster than the environment dynamics.

### Phase 8 — Domain randomization

> Goal: Make policies robust to the sim-to-real gap.

- [ ] **8.1** Add noise injection config to `BaseEnv` — Gaussian noise on state observations, configurable per-dimension.
- [ ] **8.2** Add MJCF randomization hooks — environments can vary physics parameters (mass, friction, damping, actuator gain) each episode by modifying the MJCF before `env.reset()`.
- [ ] **8.3** Test that a domain-randomized policy is more robust to perturbation than a standard one.

### Phase 9 — Advanced algorithms

> Goal: Move beyond vanilla DQN for harder problems.

- [ ] **9.1** Double DQN (decouple action selection from value estimation to reduce overestimation bias).
- [ ] **9.2** Prioritized experience replay (sample important transitions more often).
- [ ] **9.3** Dueling DQN (separate state-value and advantage streams).
- [ ] **9.4** Evaluate whether continuous-action algorithms (SAC, TD3) should replace DQN for MuJoCo environments where action discretization loses too much fidelity.

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

This is what `eval.py` does. An export step (Phase 6) would convert this dictionary into a self-contained format that doesn't need Python or the config to run.
