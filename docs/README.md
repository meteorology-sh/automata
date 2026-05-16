## Data Structures

### Images as Data

An image is a three-dimensional array of numbers, having a height, width, and channels. These channels are arrays with values for blue, green, and red.

The first step is commonly converting the channels into a single channel, in grayscale. Next, the images are downsized, depending on what kinds of outcomes are desirable. A roomba may need less dimensionality, whereas a drone may need more. The size of the image corresponds to the quality of resolution and data interpretable by the program.

We can train existing models by fine tuning on specific data, in a process called transfer learning.

Further, images are normalized, so the pixel values are transformed from 0-255 to 0 to 1.

- An image is a (H, W, 3) array of pixel values
- Grayscale collapses the channels to (H, W)
- We resize to a fixed input size the model expects
- We normalize to 0–1 to make training stable
- Color vs grayscale is a deliberate choice based on the task
- Images are analyzed in batches, giving input matrices a rank 4

### Vectors, Matrices, and Tensors

- A single number → scalar (rank 0)
- A 1D array → vector (rank 1)
- A 2D array → matrix (rank 2)
- A 3D+ array → tensor (rank 3+)

[batch, channel, height, width]

### Convolutional Neural Networks

A CNN is the mechanism through which raw pixels are categorized according to some ontology. A convolution slides a small grid -- called a filter or kernel -- across an image, and at each step determines whether a pattern exists. The output is a feature map.

The network learns these filters during training. Early layers learn simply things like edgesss and corners, whereass deeper layers combine them into complex things like "car", or "house".

That hierarchy looks something like:

- Layer 1 → edges, color gradients
- Layer 2 → shapes, corners
- Layer 3+ → objects, structures

### Logits

The output of a CNN are raw scores, called logits, that can be any value, positive or negative. We turn them into probabilities that sum to 1 by passing them into a softmax function.

## Reinforcement Learning

The core idea is that the drone acts in a loop:

```
observe state → take action → receive reward → observe new state → repeat
```

For a drone:

- State — the current camera frame (our image tensor)
- Action — something like move forward, turn left, turn right, ascend, descend
- Reward — a score we define, e.g. +1 for covering new ground, -10 for hitting something
- Episode — one full run, from takeoff to crash or landing

The goal of the agent is to learn a policy: A strategy for choosing actions that maximizes the total reward over time. This training takes place in a virtual environment, like `Gymnasium` or `PyBullet` or `AirSim`.

### Training

During training, the policy is guided by something called the `Q-value`. A Q-value answerss the question, given the current state, how much total reward can the model expect if it takes action.

```
Q(state, action) → expected future reward
```

The drone takes the action with the highest Q-value. That's the policy. The agent has to learn the rules for Q-values through training, and it does so through an update rule called the Bellman equation:

```
Q(state, action) = reward + (discount × best Q-value in next state)
```

The discount is usually _0.99_, but otherwise a number between 0 and 1 that makes the agent slightly prefer immediate rewards over distant ones. This is essential because avoiding a collision is an immediate and higher priority goal than whatever the drone was set out to do in the first place.

The learning loop therefore takes the following form:

- Observe state
- Pick action with highest Q-value
- Receive reward, land in new state
- Update Q-values using the Bellman equation
- Repeat

### Epsilon Greed

If an agent always picks the action with the highest Q-value, it will only ever exploit options that it already knows, and not learn new things. The solution is exploration versus exploitation, controlled by epsilon greed.

- With probability epsilon (ε) — take a random action (explore)
- With probability 1 - ε — take the best known action (exploit)

The training session starts with ε close to 1 (almost fully random) and decay it over time as the agent builds up knowledge. Early on the drone flails around randomly, gradually becoming more deliberate as it learns which actions lead to good rewards.

In summary, training looks like this:

- The agent/environment/state/action/reward loop
- Q-values as a measure of action quality
- The Bellman equation as the update rule
- Epsilon-greedy as the exploration strategy

## Deep Q-Networks (DQN)

The insight behind a DQN is that we can use a CNN to approximate Q-values instead of a table. Given an image, we feed it through a CNN that outputs a Q-value for every possibible action simultaneously.

The DQN doesn't apply the softmax anymore, because we don't want to normalize the options. In fact we want to know the relative value of our options, like turning left versus right.

### Replay Buffer

The replay buffer is a random sample of other experiences that introduces a diverse range of experiences during its training regimen, rather than its sequential order. It's like flash cards.

## Reward Shaping and Training

The core challenge is the reward signal is the only way you communicate intentions to the agent. There is some parallel here to prompt engineering: If you're not careful, the agent will get creative to serve its goals.

The broader lesson is that reward shaping requires you to think adversarially about your own reward function -- asking "How would a completely amoral optimizer try to game this?" Whatever answer you come up with, the agent will probably find it eventually.

The three practical stopping criteria are:

1. Performance threshold — average reward consistently above some target
2. Diminishing returns — reward has plateaued and isn't improving
3. Budget — you've hit your compute or time limit and take the best model so far
