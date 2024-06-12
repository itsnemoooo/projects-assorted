import numpy as np

class Random:
    def __init__(self, name, number_of_arms):
        self._number_of_arms = number_of_arms
        self.name = name

    def step(self, unused_previous_action, unused_reward):
        return np.random.randint(self._number_of_arms)

    def reset(self):
        pass

class EpsilonGreedy:
    def __init__(self, name, number_of_arms, epsilon=0.1):
        self.name = name
        self._number_of_arms = number_of_arms
        self.epsilon = epsilon
        self.reset()

    def step(self, previous_action, reward):
        if callable(self.epsilon):
            epsilon_val = self.epsilon(self._total_counts + 1e-9)
        else:
            epsilon_val = self.epsilon

        if np.random.rand() < epsilon_val:
            action = np.random.randint(self._number_of_arms)
        else:
            max_value = np.max(self._action_values)
            max_actions = np.where(self._action_values == max_value)[0]
            action = np.random.choice(max_actions)

        if previous_action is not None:
            self._counts[previous_action] += 1
            n = self._counts[previous_action]
            value = self._action_values[previous_action]
            self._action_values[previous_action] += (reward - value) / n
            self._total_counts += 1

        return action

    def reset(self):
        self._counts = np.zeros(self._number_of_arms)
        self._action_values = np.zeros(self._number_of_arms)
        self._total_counts = 0

class UCB:
    def __init__(self, name, number_of_arms, bonus_multiplier):
        self._number_of_arms = number_of_arms
        self._bonus_multiplier = bonus_multiplier
        self.name = name
        self.reset()

    def step(self, previous_action, reward):
        if previous_action is not None:
            self.counts[previous_action] += 1
            self.total_rewards[previous_action] += reward

        self.t += 1
        avg_rewards = np.zeros(self._number_of_arms)
        ucb_values = np.zeros(self._number_of_arms)
        for arm in range(self._number_of_arms):
            if self.counts[arm] > 0:
                avg_rewards[arm] = self.total_rewards[arm] / self.counts[arm]
                confidence = self._bonus_multiplier * np.sqrt(np.log(self.t) / self.counts[arm])
                ucb_values[arm] = avg_rewards[arm] + confidence
            else:
                ucb_values[arm] = float('inf')

        action = np.argmax(ucb_values)

        return action

    def reset(self):
        self.counts = np.zeros(self._number_of_arms, dtype=int)
        self.total_rewards = np.zeros(self._number_of_arms)
        self.t = 0

class REINFORCE:
    def __init__(self, name, number_of_arms, step_size=0.1, baseline=False):
        self.name = name
        self.number_of_arms = number_of_arms
        self.step_size = step_size
        self.baseline = baseline
        self.preferences = np.zeros(number_of_arms) + 0.1
        self.average_reward = 0.0
        self.total_steps = 0

    def step(self, previous_action, reward):
        reward = 0 if reward is None else reward

        self.total_steps += 1

        baseline = self.average_reward if self.baseline else 0

        squared_preferences = np.square(self.preferences)
        sum_p_b = np.sum(squared_preferences)

        for a in range(self.number_of_arms):
            if a == previous_action:
                grad = 2 * (self.preferences[a]) * (1/(self.preferences[a])**2 - 1/(sum_p_b))
            else:
                grad = -2 * self.preferences[a]/(sum_p_b)
            self.preferences[a] += self.step_size * (reward - baseline) * grad

        squared_preferences = np.square(self.preferences)
        sum_p_b = np.sum(squared_preferences)

        squaremax_probs = squared_preferences / sum_p_b

        if self.baseline:
            self.average_reward += self.step_size * (reward - self.average_reward)

        next_action = np.random.choice(self.number_of_arms, p=squaremax_probs)
        return next_action

    def reset(self):
        self.preferences = np.zeros(self.number_of_arms) + 0.1
        self.average_reward = 0.0
        self.total_steps = 0
