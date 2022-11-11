"""Game class to represent 2048 game state."""

import numpy as np
import torch
import math
from collections import Counter

ACTION_NAMES = ["left", "up", "right", "down"]
ACTION_LEFT = 0
ACTION_UP = 1
ACTION_RIGHT = 2
ACTION_DOWN = 3


class Game(object):
    """Represents a 2048 Game state and implements the actions.
    Implements the 2048 Game logic, as specified by this source file:
    https://github.com/gabrielecirulli/2048/blob/master/js/game_manager.js
    Game states are represented as shape (4, 4) numpy arrays whos entries are 0
    for empty fields and ln2(value) for any tiles.
    """

    def __init__(self, state=None, initial_score=0):
        """Init the Game object.
        Args:
          state: Shape (4, 4) numpy array to initialize the state with. If None,
              the state will be initialized with with two random tiles (as done
              in the original game).
          initial_score: Score to initialize the Game with.
        """

        self._score = initial_score

        if state is None:
            self._state = np.zeros((4, 4), dtype=np.int)
            self.add_random_tile()
            self.add_random_tile()
        else:
            self._state = state

    def copy(self):
        """Return a copy of self."""

        return Game(np.copy(self._state), self._score)

    def game_over(self):
        """Whether the game is over."""
        # if self.max_tile() >= 2048:  # stop at winning state
        #     return True

        for action in range(4):
            if self.is_action_available(action):
                return False
        return True

    def available_actions(self):
        """Computes the set of actions that are available."""
        return [action for action in range(4) if self.is_action_available(action)]

    def is_action_available(self, action):
        """Determines whether action is available.
        That is, executing it would change the state.
        """

        temp_state = np.rot90(self._state, action)
        return self._is_action_available_left(temp_state)

    def _is_action_available_left(self, state):
        """Determines whether action 'Left' is available."""

        # True if any field is 0 (empty) on the left of a tile or two tiles can
        # be merged.
        for row in range(4):
            has_empty = False
            for col in range(4):
                has_empty |= state[row, col] == 0
                if state[row, col] != 0 and has_empty:
                    return True
                if (state[row, col] != 0 and col > 0 and
                        state[row, col] == state[row, col - 1]):
                    return True

        return False

    def do_action(self, action):
        """Execute action, add a new tile, update the score & return the reward."""

        if not self.is_action_available(action):
            return 0
        temp_state = np.rot90(self._state, action)
        reward = self._do_action_left(temp_state)
        self._state = np.rot90(temp_state, -action)
        self._score += reward

        self.add_random_tile()

        return reward

    def _do_action_left(self, state):
        """Exectures action 'Left'."""

        reward = 0

        for row in range(4):
            # Always the rightmost tile in the current row that was already moved
            merge_candidate = -1
            merged = np.zeros((4,), dtype=np.bool)

            for col in range(4):
                if state[row, col] == 0:
                    continue

                if (merge_candidate != -1 and
                        not merged[merge_candidate] and
                        state[row, merge_candidate] == state[row, col]):
                    # Merge tile with merge_candidate
                    state[row, col] = 0
                    merged[merge_candidate] = True
                    state[row, merge_candidate] += 1
                    reward += 2 ** state[row, merge_candidate]

                else:
                    # Move tile to the left
                    merge_candidate += 1
                    if col != merge_candidate:
                        state[row, merge_candidate] = state[row, col]
                        state[row, col] = 0

        return reward

    def add_random_tile(self):
        """Adds a random tile to the grid. Assumes that it has empty fields."""

        x_pos, y_pos = np.where(self._state == 0)
        assert len(x_pos) != 0
        empty_index = np.random.choice(len(x_pos))
        value = np.random.choice([1, 2], p=[0.9, 0.1])

        self._state[x_pos[empty_index], y_pos[empty_index]] = value

    def print_state(self):
        """Prints the current state."""

        def tile_string(value):
            """Concert value to string."""
            if value > 0:
                return '% 5d' % (2 ** value,)
            return "     "

        print("=" * 50)
        print("-" * 25)
        for row in range(4):
            print("|" + "|".join([tile_string(v) for v in self._state[row, :]]) + "|")
            print("-" * 25)
        print("=" * 50)

    def state(self):
        """Return current state."""
        return self._state

    def score(self):
        """Return current score."""
        return self._score

    def vector(self):
        vec = torch.zeros(256)
        for i, num in enumerate(np.array(self._state).flatten()):
            vec[i * 16 + num - 1] = 1
        return vec

    def action_available(self):
        return np.array([self.is_action_available(action) for action in range(4)])

    def max_tile(self):
        return 2 ** np.max(self._state)

    def get_next_state(self, action):
        new_state = self._state.copy()
        temp_state = np.rot90(new_state, action)
        reward = self._do_action_left(temp_state)
        new_state = np.rot90(temp_state, -action)
        vec = torch.zeros(256)
        for i, num in enumerate(np.array(new_state).flatten()):
            vec[i * 16 + num - 1] = 1
        if not self.is_action_available(action):
            return new_state, vec, 0
        return new_state, vec, reward

    def inbound(self, c):
        return 0 <= c[0] < 4 and 0 <= c[1] < 4

    def findFarthestPosition(self, cell, vector):
        while True:
            previous = cell
            cell = (cell[0] + vector[0], cell[1] + vector[1])
            if not self.inbound(cell) or self._state[cell[0], cell[1]] != 0:
                break
        return previous, cell

    def smoothness(self):
        vectors = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        smoothness = 0
        for x in range(4):
            for y in range(4):
                value = self._state[x, y]
                for d in [0, 1, 2, 3]:
                    v = vectors[d]
                    targetCell = self.findFarthestPosition((x, y), v)[1]
                    target = self._state[targetCell[0], targetCell[1]] if self.inbound(targetCell) else 0
                    if target != 0:
                        smoothness -= abs(value - target)
        return smoothness

    def monotonicity(self):
        totals = [0, 0, 0, 0]
        for x in range(4):
            current = 0
            n = current + 1
            while n < 4:
                while n < 4 and self._state[x, n] == 0:
                    n += 1
                if n >= 4: n -= 1
                cur_value = self._state[x, current]
                next_value = self._state[x, n]
                if cur_value > next_value:
                    totals[0] += next_value - cur_value
                else:
                    totals[1] += cur_value - next_value
                current = n
                n += 1

        for y in range(4):
            current = 0
            n = current + 1
            while n < 4:
                while n < 4 and self._state[n, y] == 0: n += 1
                if n >= 4: n -= 1
                cur_value = self._state[current, y]
                next_value = self._state[n, y]
                if cur_value > next_value:
                    totals[2] += next_value - cur_value
                else:
                    totals[3] += cur_value - next_value
                current = n
                n += 1
        return max(totals[0], totals[1]) + max(totals[2], totals[3])

    def uniformity(self):
        uniformity = 0
        values = []
        for row in range(4):
            for col in range(4):
                values.append(self._state[row, col])
        count = Counter(values)

        for key in count:
            uniformity += (count[key]) ** 3

        return uniformity

    def number_of_potential_merges(self):
        merges = 0
        for row in range(4):
            for col in range(4):
                if self.inbound([row, col + 1]):
                    if self._state[row, col + 1] == self._state[row, col]:
                        merges += 1
                if self.inbound([row + 1, col]):
                    if self._state[row + 1, col] == self._state[row, col]:
                        merges += 1

        return merges

    def eval_row(self, board, row_index):
        '''
        calculate evaluation score for a single row
        '''
        row = board[row_index]

        score_lost_penalty = 0
        score_monotonicity_power = 4
        score_monotonicity_weight = 47
        score_sum_power = 3.5
        score_sum_weight = 0
        score_merges_weight = 700
        score_empty_weight = 270

        empty = np.sum(row == 0)

        merges = 0  # number of tiles that can be merged
        prev = 0
        counter = 0
        total = 0  # sum
        for i in range(4):
            rank = row[i]
            total += pow(rank, score_sum_power)
            if rank != 0:
                if prev == rank:
                    counter += 1
                elif counter > 0:
                    merges += 1 + counter
                    counter = 0
                prev = rank

        if counter > 0:
            merges += 1 + counter

        monotonicity_left = 20000
        monotonicity_right = 0
        for i in range(1, 4):
            if row[i - 1] > row[i]:
                monotonicity_left += pow(row[i - 1], score_monotonicity_power) - pow(row[i], score_monotonicity_power)
            else:
                monotonicity_right += pow(row[i], score_monotonicity_power) - pow(row[i - 1], score_monotonicity_power)

        # return score_lost_penalty + score_empty_weight * empty + score_merges_weight * merges \
        #        - score_monotonicity_weight * min(monotonicity_left, monotonicity_right) - score_sum_weight * total

        return score_empty_weight * empty + score_merges_weight * merges

    # def eval(self):
    #     score = 0
    #     # add evaluation score for all rows
    #     for row_index in range(4):
    #         score += self.eval_row(self._state, row_index)
    #
    #     temp_board = np.rot90(self._state, 1)
    #     # add evaluation score for all columns
    #     for row_index in range(4):
    #         score += self.eval_row(temp_board, row_index)
    #
    #     return score

    def eval(self):
        emptyCells = np.sum(self._state == 0)
        emptyCellScore = math.log(emptyCells) if emptyCells > 0 else 0
        smoothWeight = 0.1
        monoWeight = 1.0
        emptyWeight = 2.7
        maxWeight = 1.0
        return self.smoothness() * smoothWeight + self.monotonicity() * monoWeight + emptyCellScore * emptyWeight + np.max(
            self._state) * maxWeight
        #return self.number_of_potential_merges() + 0.5 * emptyCells






# Use single heuristic results:
# Note: with only monotonicity, 1024 can be reached 8/10, 2048 can be reached 0/10
#       with only emptyScore,   1024 can be reached 10/10, 2048 can be reached 1/10
#      *with only # merges,     1024 can be reached 8/10, 2048 can be reached 5/10     -- my code
#       with only smoothness,   1024 can be reached 1/10, 2048 can be reached 0/10     -- original code
#       with only max state,    1024 can be reached 0/10, 2048 can be reached 0/10

# Use multiple heuristics:
#
