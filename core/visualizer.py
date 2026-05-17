import threading
import matplotlib.pyplot as plt
from collections import deque


class Visualizer:
    """
    Real-time training dashboard. Runs in a background thread so it never blocks training.

    Displays:
      - Episode reward over time (with solve threshold and rolling average)
      - Current epsilon value

    Two update methods:
      - update()       — called every step (currently unused, kept for interface compatibility)
      - end_episode()  — called once per episode with total reward and epsilon
    """

    def __init__(self, num_actions: int = 0, action_labels: list = None,
                 maxlen: int = 500, solve_threshold: float = None,
                 solve_window: int = 50):
        self.maxlen = maxlen
        self.solve_threshold = solve_threshold
        self.solve_window = solve_window

        self._episode_rewards = deque(maxlen=maxlen)
        self._episode_count = 0
        self._latest_epsilon = 1.0
        self._lock = threading.Lock()
        self._running = False

    def update(self, state, q_values: list):
        """Called by the training loop at each step. Thread-safe."""
        pass

    def end_episode(self, total_reward: float, epsilon: float):
        """Called by the training loop at the end of each episode. Thread-safe."""
        with self._lock:
            self._episode_rewards.append(total_reward)
            self._episode_count += 1
            self._latest_epsilon = epsilon

    def start(self):
        """Start the dashboard in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _run(self):
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("#1e1e1e")
        ax.set_facecolor("#2d2d2d")
        ax.tick_params(colors="#cccccc")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        while self._running:
            with self._lock:
                ep_rewards = list(self._episode_rewards)
                ep_count = self._episode_count
                epsilon = self._latest_epsilon

            if not ep_rewards:
                plt.pause(1.0)
                continue

            n = len(ep_rewards)
            episodes_x = list(range(ep_count - n, ep_count))

            ax.cla()
            ax.set_facecolor("#2d2d2d")
            ax.plot(episodes_x, ep_rewards, color="#4fc3f7", linewidth=1, alpha=0.4, label="Episode")

            # Rolling average — matches solve condition window
            if n >= 5:
                window = min(self.solve_window, n)
                avg = [sum(ep_rewards[max(0, i - window + 1):i + 1]) / min(i + 1, window)
                       for i in range(n)]
                ax.plot(episodes_x, avg, color="#4fc3f7", linewidth=2, label=f"Avg ({window})")

            # Solve threshold reference line
            if self.solve_threshold is not None:
                ax.axhline(y=self.solve_threshold, color="#66bb6a",
                           linewidth=1, linestyle="--", alpha=0.7, label="Solve")

            ax.set_title(f"Episode reward        \u03b5 = {epsilon:.4f}",
                         color="#cccccc", fontsize=11)
            ax.set_xlabel("Episode", color="#cccccc", fontsize=9)
            ax.legend(loc="upper left", fontsize=8, facecolor="#2d2d2d",
                      edgecolor="#444444", labelcolor="#cccccc")
            ax.tick_params(colors="#cccccc")

            plt.tight_layout()
            plt.pause(1.0)

        plt.ioff()
        plt.close()
