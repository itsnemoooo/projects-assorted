import numpy as np

class BernoulliBandit:
    """A stationary multi-armed Bernoulli bandit."""

    def __init__(self, success_probabilities, success_reward=1., fail_reward=0.):
        self._probs = success_probabilities
        self._number_of_arms = len(self._probs)
        self._s = success_reward
        self._f = fail_reward

        ps = np.array(success_probabilities)
        self._values = ps * success_reward + (1 - ps) * fail_reward

    def step(self, action):
        if action < 0 or action >= self._number_of_arms:
            raise ValueError('Action {} is out of bounds for a {}-armed bandit'.format(action, self._number_of_arms))
        success = bool(np.random.random() < self._probs[action])
        reward = success * self._s + (not success) * self._f
        return reward

    def regret(self, action):
        return self._values.max() - self._values[action]

    def optimal_value(self):
        return self._values.max()

class NonStationaryBandit:
    """A non-stationary multi-armed Bernoulli bandit."""

    def __init__(self, success_probabilities, success_reward=1., fail_reward=0., change_point=800, change_is_good=True):
        self._probs = success_probabilities
        self._number_of_arms = len(self._probs)
        self._s = success_reward
        self._f = fail_reward
        self._change_point = change_point
        self._change_is_good = change_is_good
        self._number_of_steps_so_far = 0

        ps = np.array(success_probabilities)
        self._values = ps * success_reward + (1 - ps) * fail_reward

    def step(self, action):
        if action < 0 or action >= self._number_of_arms:
            raise ValueError('Action {} is out of bounds for a {}-armed bandit'.format(action, self._number_of_arms))
        self._number_of_steps_so_far += 1
        success = bool(np.random.random() < self._probs[action])
        reward = success * self._s + (not success) * self._f

        if self._number_of_steps_so_far == self._change_point:
            reward_dif = (self._s - self._f)
            if self._change_is_good:
                self._f = self._s + reward_dif
            else:
                self._s -= reward_dif
                self._f += reward_dif

            ps = np.array(self._probs)
            self._values = ps * self._s + (1 - ps) * self._f

        return reward

    def regret(self, action):
        return self._values.max() - self._values[action]

    def optimal_value(self):
        return self._values.max()
