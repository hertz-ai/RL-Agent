import datetime
import os

import numpy
import torch
from create_kg import KG
import random
import json
import requests

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available
        self.max_num_gpus = None

        # Game
        # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.observation_shape = (3, 6, 7)
        # Fixed list of all possible actions. You should only edit the length
        self.action_space = list(range(7))
        # List of players. You should only edit the length
        self.players = list(range(1))  # ! I changed this
        # Number of previous observations and previous actions to add to the current observation
        self.stacked_observations = 0

        # Evaluate
        # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.muzero_player = 0
        self.opponent = None  # ! I changed this
        # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        """Self-Play"""
        # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.num_workers = 1
        self.selfplay_on_gpu = False
        self.max_moves = 42  # Maximum number of moves if game is not finished before
        self.num_simulations = 200  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time
        self.temperature_threshold = None

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        """UCB formula"""
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        """Network"""
        self.network = "resnet"  # "resnet" / "fullyconnected"
        # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        self.support_size = 10

        """Residual Network"""
        # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.downsample = False
        self.blocks = 3  # Number of blocks in the ResNet
        self.channels = 64  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_reward_layers = [64]
        # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_value_layers = [64]
        # Define the hidden layers in the policy head of the prediction network
        self.resnet_fc_policy_layers = [64]

        """Fully Connected Network"""
        self.encoding_size = 32
        # Define the hidden layers in the representation network
        self.fc_representation_layers = []
        # Define the hidden layers in the dynamics network
        self.fc_dynamics_layers = [64]
        # Define the hidden layers in the reward network
        self.fc_reward_layers = [64]
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network

        """Training"""
        self.results_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../results",
            os.path.basename(__file__)[:-3],
            datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"),
        )  # Path to store the model weights and TensorBoard logs
        self.save_model = (
            True  # Save the checkpoint in results_path as model.checkpoint
        )
        # Total number of training steps (ie weights update according to a batch)
        self.training_steps = 100000
        self.batch_size = (
            64  # Number of parts of games to train on at each training step
        )
        # Number of training steps before using the model for self-playing
        self.checkpoint_interval = 10
        # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.value_loss_weight = 0.25
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.005  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000

        """ Replay Buffer """
        # Number of self-play games to keep in the replay buffer
        self.replay_buffer_size = 10000
        self.num_unroll_steps = (
            42  # Number of game moves to keep for every batch element
        )
        # Number of steps in the future to take into account for calculating the target value
        self.td_steps = 42
        # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER = True
        # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1
        self.PER_alpha = 0.5

        # Reanalyze (See paper appendix Reanalyse)
        # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.use_last_model_value = True
        self.reanalyse_on_gpu = False

        # Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Student()

    def step(self, action):
        # * Compulsory
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward * 10, done

    # def to_play(self):
    #     """
    #     Return the current player.

    #     Returns:
    #     # ! don't need it
    #         The current player, it should be an element of the players list in the config.
    #     """
    #     return self.env.to_play()

    def legal_actions(self):
        # * Compulsory
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        # * Compulsory
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        # * Compulsory
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    # def human_to_action(self):
    #     """
    #     For multiplayer games, ask the user for a legal action
    #     and return the corresponding action number.

    #     Returns:
    #         An integer from the action space.
    #     """
    #     # ! dont need it
    #     choice = input(
    #         f"Enter the column to play for the player {self.to_play()}: ")
    #     while choice not in [str(action) for action in self.legal_actions()]:
    #         choice = input("Enter another column : ")
    #     return int(choice)

    # def expert_agent(self):
    #     """
    #     Hard coded agent that MuZero faces to assess his progress in multiplayer games.
    #     It doesn't influence training

    #     Returns:
    #         Action as an integer to take in the current game state
    #     """
    #     # ! dont need it
    #     return self.env.expert_action()

    def action_to_string(self, action_vector):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        response = str(self.env.node_sequence) + "\n" + str(action_vector)
        return response


class Student:
    @staticmethod
    def randomlyIncreased(score: float) -> float:
        """
        increases the score by a small random amount. the increase is inversely proportional to the original score.

        Args:
            score (float): the original score

        Returns:
            float: the original score increased by a small amount.
        """
        return score + random.triangular(0, 1 - score, 0.01 * (1 - score))

    def __init__(self):
        self.KG = KG()
        self.KG.makeGraph()
        self.KG.pruneGraph()
        self.KG.makeAdjacencyMatrix()
        self.KG.initializeScores()

        self.n_concepts = self.KG.n_nodes
        self.node_sequence = self.KG.G.nodes()
        self.state = {
            "knowledge": [node["knowledge score"] for node in self.KG.G.nodes()],
            "application": [node["application score"] for node in self.KG.G.nodes()],
        }
        self.knowledge_graph = self.KG.adjMatrix
        self.goal_state = {
            "knowledge": [
                self.randomlyIncreased(node["knowledge score"])
                for node in self.KG.G.nodes()
            ],
            "application": [
                self.randomlyIncreased(node["application score"])
                for node in self.KG.G.nodes()
            ],
        }

    # def to_play(self):
    #     # ! commented in Game class too
    #     return 0 if self.player == 1 else 1

    def reset(self):
        """
        for next / new student.
        """
        self.KG.resetScores()
        self.KG.initializeScores()
        self.n_concepts = self.KG.n_nodes
        self.node_sequence = self.KG.G.nodes()
        self.state = {
            "knowledge": [node["knowledge score"] for node in self.KG.G.nodes()],
            "application": [node["application score"] for node in self.KG.G.nodes()],
        }
        self.goal_state = {
            "knowledge": [
                self.randomlyIncreased(node["knowledge score"])
                for node in self.KG.G.nodes()
            ],
            "application": [
                self.randomlyIncreased(node["application score"])
                for node in self.KG.G.nodes()
            ],
        }
        self.knowledge_graph = self.KG.adjMatrix

    def step(self, action):
        def get_materials(action):
            # ! dummy function
            return [0]

        def generate_questions(material):
            body = {
                "text_corpus": material,
                "assessment": {
                    "name": "".join(
                        [
                            random.choice(string.ascii_letters)
                            for _ in random.randint(1, 15)
                        ]
                    ),
                    "course_id": 21,
                    "number_of_questions": 1,
                    "assessment_type": "MCQ",
                    "is_active": True,
                },
            }
            response = requests.post("", data=json.dumps(body), headers={'Content-Type': 'application/json'})
            return [0]

        def get_answers(question, material):
            # ! dummy function
            return question + material

        def get_scores(question, answer):
            # ! dummy function
            return question + answer

        """
        our action is a weighted vector of length self.n_concepts
        reward is the formula and the score in the test
        done is if the goal is reached
        - or the user is within some threshold of the goal
        """

        materials = get_materials(action)
        questions = [generate_questions(material) for material in materials]
        # format is incorrect for the below question
        answers = [
            get_answers(question, material)
            for question, material in zip(questions, materials)
        ]
        scores = [
            get_scores(question, answer) for question, answer in zip(questions, answers)
        ]

        reward = 0
        done = 0
        return self.get_observation(), reward, done

    def get_observation(self):
        board_player1 = numpy.where(self.board == 1, 1.0, 0.0)
        board_player2 = numpy.where(self.board == -1, 1.0, 0.0)
        board_to_play = numpy.full((6, 7), self.player, dtype="int32")
        return numpy.array([board_player1, board_player2, board_to_play])

    def legal_actions(self):
        legal = []
        for i in range(7):
            if self.board[5][i] == 0:
                legal.append(i)
        return legal

    def have_winner(self):
        # Horizontal check
        for i in range(4):
            for j in range(6):
                if (
                    self.board[j][i] == self.player
                    and self.board[j][i + 1] == self.player
                    and self.board[j][i + 2] == self.player
                    and self.board[j][i + 3] == self.player
                ):
                    return True

        # Vertical check
        for i in range(7):
            for j in range(3):
                if (
                    self.board[j][i] == self.player
                    and self.board[j + 1][i] == self.player
                    and self.board[j + 2][i] == self.player
                    and self.board[j + 3][i] == self.player
                ):
                    return True

        # Positive diagonal check
        for i in range(4):
            for j in range(3):
                if (
                    self.board[j][i] == self.player
                    and self.board[j + 1][i + 1] == self.player
                    and self.board[j + 2][i + 2] == self.player
                    and self.board[j + 3][i + 3] == self.player
                ):
                    return True

        # Negative diagonal check
        for i in range(4):
            for j in range(3, 6):
                if (
                    self.board[j][i] == self.player
                    and self.board[j - 1][i + 1] == self.player
                    and self.board[j - 2][i + 2] == self.player
                    and self.board[j - 3][i + 3] == self.player
                ):
                    return True

        return False

    def expert_action(self):
        board = self.board
        action = numpy.random.choice(self.legal_actions())
        for k in range(3):
            for l in range(4):
                sub_board = board[k : k + 4, l : l + 4]
                # Horizontal and vertical checks
                for i in range(4):
                    if abs(sum(sub_board[i, :])) == 3:
                        ind = numpy.where(sub_board[i, :] == 0)[0][0]
                        if numpy.count_nonzero(board[:, ind + l]) == i + k:
                            action = ind + l
                            if self.player * sum(sub_board[i, :]) > 0:
                                return action

                    if abs(sum(sub_board[:, i])) == 3:
                        action = i + l
                        if self.player * sum(sub_board[:, i]) > 0:
                            return action
                # Diagonal checks
                diag = sub_board.diagonal()
                anti_diag = numpy.fliplr(sub_board).diagonal()
                if abs(sum(diag)) == 3:
                    ind = numpy.where(diag == 0)[0][0]
                    if numpy.count_nonzero(board[:, ind + l]) == ind + k:
                        action = ind + l
                        if self.player * sum(diag) > 0:
                            return action

                if abs(sum(anti_diag)) == 3:
                    ind = numpy.where(anti_diag == 0)[0][0]
                    if numpy.count_nonzero(board[:, 3 - ind + l]) == ind + k:
                        action = 3 - ind + l
                        if self.player * sum(anti_diag) > 0:
                            return action

        return action

    def render(self):
        print(self.board[::-1])
