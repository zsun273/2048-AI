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
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        self.width = 4
        self.height = 4
        self.th = 128
        self.init_board(state)
        # if state is None:
        #     self._state = np.zeros((4, 4), dtype=np.int)
        #     self.add_random_tile()
        #     self.add_random_tile()
        # else:
        #     self._state = state

    def init_board(self, state=None):
        # keep available moves in a list
        # self.states = {}
        # self.last_move = -1
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
        """Whether the game is over.
            return end: if end
                    winning: win or lose
        """
        if self.max_tile() >= self.th:  # stop at winning state
            return True, 1

        for action in range(4):
            if self.is_action_available(action):
                return False, -1
        return True, -1

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

    def do_move(self, action):
        """Execute action, without adding a new tile."""

        if not self.is_action_available(action):
            return 0
        temp_state = np.rot90(self._state, action)
        self._do_action_left(temp_state)
        self._state = np.rot90(temp_state, -action)


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

        return self.to_string()

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

    def to_string(self):
        vec = ["0"]*16
        for i, num in enumerate(np.array(self._state).flatten()):
            vec[i] = str(num)
        return "".join(vec)

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


    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 1*width*height
        """
        square_state = self._state.reshape(-1, self.width, self.height)

        # if self.states:
        #     moves, players = np.array(list(zip(*self.states.items())))
        #     move_curr = moves[players == self.current_player]
        #     move_oppo = moves[players != self.current_player]
        #     square_state[0][move_curr // self.width,
        #                     move_curr % self.height] = 1.0
        #     square_state[1][move_oppo // self.width,
        #                     move_oppo % self.height] = 1.0
        #     # indicate the last move location
        #     square_state[2][self.last_move // self.width,
        #                     self.last_move % self.height] = 1.0
        # if len(self.states) % 2 == 0:
        #     square_state[3][:, :] = 1.0  # indicate the colour to play
        return np.copy(square_state)

class Play:
    """2048 game server"""
    def __init__(self, game, **kwargs):
        self.board = game


    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        states, mcts_probs, winning = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)

            # perform a move
            string_board = self.board.to_string()
            self.board.do_move(move)
            self.board.add_random_tile()
            player.mcts.update_with_status(string_board, move)
            end, winner = self.board.game_over()
            if end:
                # winner from the perspective of the current player of each state
                self.board.print_state()
                if winner == 1:
                    winners_z = np.ones(len(states), dtype=float)
                else:
                    winners_z = np.full(100, -1.0)
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)