import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque


class Visualizer:
    """
    Real-time training dashboard. Runs in a background thread so it never blocks training.

    Displays:
      - Episode reward over time
      - Current epsilon
      - Q-values for each action at the current step
      - Live agent view (image frame or vector bar chart)

    Usage:
      visualizer = Visualizer(num_actions=2, action_labels=["left", "right"])
      visualizer.start()
      # pass to train() — it calls visualizer.update() each step
      visualizer.stop()
    """

    def __init__(self, num_actions: int, action_labels: list = None, maxlen: int = 500):
        self.num_actions = num_actions
        self.action_labels = action_labels or [
            f"a{i}" for i in range(num_actions)]
        self.maxlen = maxlen

        self._rewards = deque(maxlen=maxlen)
        self._epsilons = deque(maxlen=maxlen)
        self._latest_state = None
        self._latest_q = None
        self._lock = threading.Lock()
        self._running = False

    def update(self, state, q_values: list, reward: float, epsilon: float, loss=None):
        """Called by the training loop at each step. Thread-safe."""
        with self._lock:
            self._rewards.append(reward)
            self._epsilons.append(epsilon)
            self._latest_state = state
            self._latest_q = q_values

    def start(self):
        """Start the dashboard in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _run(self):
        plt.ion()
        fig = plt.figure(figsize=(12, 7))
        fig.patch.set_facecolor("#1e1e1e")
        gs = gridspec.GridSpec(2, 2, figure=fig)

        ax_reward = fig.add_subplot(gs[0, 0])
        ax_epsilon = fig.add_subplot(gs[0, 1])
        ax_qvals = fig.add_subplot(gs[1, 0])
        ax_state = fig.add_subplot(gs[1, 1])

        for ax in [ax_reward, ax_epsilon, ax_qvals, ax_state]:
            ax.set_facecolor("#2d2d2d")
            ax.tick_params(colors="#cccccc")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")

        while self._running:
            with self._lock:
                rewards = list(self._rewards)
                epsilons = list(self._epsilons)
                q_values = self._latest_q
                state = self._latest_state

            if not rewards:
                plt.pause(0.1)
                continue

            # Episode reward
            ax_reward.cla()
            ax_reward.set_facecolor("#2d2d2d")
            ax_reward.plot(rewards, color="#4fc3f7", linewidth=1)
            ax_reward.set_title("Reward per step",
                                color="#cccccc", fontsize=10)
            ax_reward.set_xlabel("Step", color="#cccccc", fontsize=8)
            ax_reward.tick_params(colors="#cccccc")

            # Epsilon
            ax_epsilon.cla()
            ax_epsilon.set_facecolor("#2d2d2d")
            ax_epsilon.plot(epsilons, color="#ce93d8", linewidth=1)
            ax_epsilon.set_title("Epsilon", color="#cccccc", fontsize=10)
            ax_epsilon.set_ylim(0, 1)
            ax_epsilon.tick_params(colors="#cccccc")

            # Q-values
            if q_values is not None:
                ax_qvals.cla()
                ax_qvals.set_facecolor("#2d2d2d")
                colors = ["#ef5350" if q == max(
                    q_values) else "#78909c" for q in q_values]
                ax_qvals.bar(self.action_labels, q_values, color=colors)
                ax_qvals.set_title("Q-values (red = selected)",
                                   color="#cccccc", fontsize=10)
                ax_qvals.tick_params(colors="#cccccc")

            # Agent view
            if state is not None:
                ax_state.cla()
                ax_state.set_facecolor("#2d2d2d")
                state_arr = np.array(state)
                if state_arr.ndim == 3:
                    ax_state.imshow(state_arr)
                    ax_state.set_title(
                        "Agent view", color="#cccccc", fontsize=10)
                    ax_state.axis("off")
                else:
                    ax_state.bar(range(len(state_arr)),
                                 state_arr, color="#80cbc4")
                    ax_state.set_title(
                        "State vector", color="#cccccc", fontsize=10)
                    ax_state.tick_params(colors="#cccccc")

            plt.tight_layout()
            plt.pause(0.1)

        plt.ioff()
        plt.close()
