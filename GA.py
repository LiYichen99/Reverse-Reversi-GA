import copy
import multiprocessing
import random
import numpy as np
from dataclasses import dataclass
import time


class GA(object):
    def __init__(self, population_size, gene_len, board_size, p_mutation, p_crossover):
        assert population_size % 2 == 0
        self.population_size = population_size
        self.gene_len = gene_len
        self.board_size = board_size
        self.p_mutation = p_mutation
        self.p_crossover = p_crossover
        self.population_list = []
        self.trans_population_list = []

    def generate_random_gene(self):
        gene = "+0b" if random.randint(0, 1) else "-0b"
        for i in range(self.gene_len-1):
            gene += str(random.randint(0, 1))
        return gene

    def generate_random_board_evaluation(self):
        board_evaluation = []
        for i in range(self.board_size):
            temp_list = []
            for j in range(self.board_size):
                temp_list.append(self.generate_random_gene())
            board_evaluation.append(temp_list)
        return board_evaluation

    def generate_origin_population(self):
        self.generation_num = 0
        for i in range(self.population_size):
            black_start_board_evaluation = self.generate_random_board_evaluation()
            black_end_board_evaluation = self.generate_random_board_evaluation()
            white_start_board_evaluation = self.generate_random_board_evaluation()
            white_end_board_evaluation = self.generate_random_board_evaluation()
            self_start_action_space_evaluation = self.generate_random_gene()
            self_end_action_space_evaluation = self.generate_random_gene()
            oppo_start_action_space_evaluation = self.generate_random_gene()
            oppo_end_action_space_evaluation = self.generate_random_gene()
            self_start_unstable_evaluation = self.generate_random_gene()
            self_end_unstable_evaluation = self.generate_random_gene()
            oppo_start_unstable_evaluation = self.generate_random_gene()
            oppo_end_unstable_evaluation = self.generate_random_gene()
            specie = {"black_start_board_evaluation": black_start_board_evaluation,
                      "black_end_board_evaluation": black_end_board_evaluation,
                      "white_start_board_evaluation": white_start_board_evaluation,
                      "white_end_board_evaluation": white_end_board_evaluation,
                      "self_start_action_space_evaluation": self_start_action_space_evaluation,
                      "self_end_action_space_evaluation": self_end_action_space_evaluation,
                      "oppo_start_action_space_evaluation": oppo_start_action_space_evaluation,
                      "oppo_end_action_space_evaluation": oppo_end_action_space_evaluation,
                      "self_start_unstable _evaluation": self_start_unstable_evaluation,
                      "self_end_unstable_evaluation": self_end_unstable_evaluation,
                      "oppo_start_unstable_evaluation": oppo_start_unstable_evaluation,
                      "oppo_end_unstable_evaluation": oppo_end_unstable_evaluation}
            self.population_list.append(specie)

    def translate_population(self):
        self.trans_population_list.clear()
        for i in range(self.population_size):
            self.trans_population_list.append(self._translate_specie(self.population_list[i]))

    def _translate_specie(self, specie):
        trans_specie = copy.deepcopy(specie)

        black_start_board_evaluation = trans_specie["black_start_board_evaluation"]
        black_end_board_evaluation = trans_specie["black_end_board_evaluation"]
        white_start_board_evaluation = trans_specie["white_start_board_evaluation"]
        white_end_board_evaluation = trans_specie["white_end_board_evaluation"]
        self_start_action_space_evaluation = trans_specie["self_start_action_space_evaluation"]
        self_end_action_space_evaluation = trans_specie["self_end_action_space_evaluation"]
        oppo_start_action_space_evaluation = trans_specie["oppo_start_action_space_evaluation"]
        oppo_end_action_space_evaluation = trans_specie["oppo_end_action_space_evaluation"]
        self_start_unstable_evaluation = trans_specie["self_start_unstable_evaluation"]
        self_end_unstable_evaluation = trans_specie["self_end_unstable_evaluation"]
        oppo_start_unstable_evaluation = trans_specie["oppo_start_unstable_evaluation"]
        oppo_end_unstable_evaluation = trans_specie["oppo_end_unstable_evaluation"]

        for i in range(self.board_size):
            for j in range(self.board_size):


                black_start_board_evaluation[i][j] = int(black_start_board_evaluation[i][j], 2)
                black_end_board_evaluation[i][j] = int(black_end_board_evaluation[i][j], 2)
                white_start_board_evaluation[i][j] = int(white_start_board_evaluation[i][j], 2)
                white_end_board_evaluation[i][j] = int(white_end_board_evaluation[i][j], 2)
        self_start_action_space_evaluation = int(self_start_action_space_evaluation, 2)
        self_end_action_space_evaluation = int(self_end_action_space_evaluation, 2)
        oppo_start_action_space_evaluation = int(oppo_start_action_space_evaluation, 2)
        oppo_end_action_space_evaluation = int(oppo_end_action_space_evaluation, 2)
        self_start_unstable_evaluation = int(self_start_unstable_evaluation, 2)
        self_end_unstable_evaluation = int(self_end_unstable_evaluation, 2)
        oppo_start_unstable_evaluation = int(oppo_start_unstable_evaluation, 2)
        oppo_end_unstable_evaluation = int(oppo_end_unstable_evaluation, 2)

        trans_specie["black_start_board_evaluation"] = black_start_board_evaluation
        trans_specie["black_end_board_evaluation"] = black_end_board_evaluation
        trans_specie["white_start_board_evaluation"] = white_start_board_evaluation
        trans_specie["white_end_board_evaluation"] = white_end_board_evaluation
        trans_specie["self_start_action_space_evaluation"] = self_start_action_space_evaluation
        trans_specie["self_end_action_space_evaluation"] = self_end_action_space_evaluation
        trans_specie["oppo_start_action_space_evaluation"] = oppo_start_action_space_evaluation
        trans_specie["oppo_end_action_space_evaluation"] = oppo_end_action_space_evaluation
        trans_specie["self_start_unstable_evaluation"] = self_start_unstable_evaluation
        trans_specie["self_end_unstable_evaluation"] = self_end_unstable_evaluation
        trans_specie["oppo_start_unstable_evaluation"] = oppo_start_unstable_evaluation
        trans_specie["oppo_end_unstable_evaluation"] = oppo_end_unstable_evaluation
        return trans_specie

    def fitness(self):
        random.shuffle(self.population_list)
        self.translate_population()
        cores = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=cores-1) as p:
            win_ratio_res = p.map(self._compete, range(self.population_size))
        p.close()
        p.join()
        print(win_ratio_res)
        for i in range(self.population_size):
            self.population_list[i]["win_ratio"] = win_ratio_res[i]
        self.population_list = sorted(self.population_list, key=lambda specie: specie["win_ratio"], reverse=True)

    def _compete(self, specie_index):
        win = 0
        lose = 0
        draw = 0
        start_time = time.time()
        specie = self.trans_population_list[specie_index]
        for i in range(self.population_size):
            if i != specie_index:
                cur_color = COLOR_BLACK
                self_color = COLOR_BLACK if i % 2 == 0 else COLOR_WHITE
                oppo_color = -self_color
                game = Game(8)
                board = game.get_init_board()
                end, winner = game.check_game_end(board)
                while not end:
                    if cur_color == self_color:
                        self_ai = MinimaxSearchPlayer(game, board, self_color, 1, 5, time.time(), specie)
                        move = self_ai.get_action()
                        board, cur_color = game.get_next_board(board, move, self_color)
                    else:
                        oppo_ai = MinimaxSearchPlayer(game, board, oppo_color, 1, 5, time.time(), self.trans_population_list[i])
                        move = oppo_ai.get_action()
                        board, cur_color = game.get_next_board(board, move, oppo_color)
                    end, winner = game.check_game_end(board)
                if winner == self_color:
                    win += 1
                elif winner == oppo_color:
                    lose += 1
                else:
                    draw += 1
        win_ratio = 1.0 * win / (win + draw + lose)
        end_time = time.time()
        print(specie_index, end_time - start_time)
        return win_ratio

    def selection(self):
        self.fitness()
        self.population_list = self.population_list[:self.population_size // 2]
        self.population_size //= 2
        assert self.population_size == len(self.population_list)

    def cross_over(self):
        random.shuffle(self.population_list)
        for i in range(self.population_size):
            specie_1 = self.population_list[i]
            specie_2 = self.population_list[i-1]
            self.population_list.append(self._cross_over(specie_1, specie_2))
        self.population_size *= 2
        assert self.population_size == len(self.population_list)

    def mutation(self):
        random.shuffle(self.population_list)
        for i in range(self.population_size):
            self._mutation(self.population_list[i])

    def _cross_over(self, specie_1, specie_2):
        black_start_board_evaluation_2 = specie_2["black_start_board_evaluation"]
        black_end_board_evaluation_2 = specie_2["black_end_board_evaluation"]
        white_start_board_evaluation_2 = specie_2["white_start_board_evaluation"]
        white_end_board_evaluation_2 = specie_2["white_end_board_evaluation"]
        self_start_action_space_evaluation_2 = specie_2["self_start_action_space_evaluation"]
        self_end_action_space_evaluation_2 = specie_2["self_end_action_space_evaluation"]
        oppo_start_action_space_evaluation_2 = specie_2["oppo_start_action_space_evaluation"]
        oppo_end_action_space_evaluation_2 = specie_2["oppo_end_action_space_evaluation"]
        self_start_unstable_evaluation_2 = specie_2["self_start_unstable_evaluation"]
        self_end_unstable_evaluation_2 = specie_2["self_end_unstable_evaluation"]
        oppo_start_unstable_evaluation_2 = specie_2["oppo_start_unstable_evaluation"]
        oppo_end_unstable_evaluation_2 = specie_2["oppo_end_unstable_evaluation"]

        new_specie = copy.deepcopy(specie_1)

        for i in range(self.board_size):
            for j in range(self.board_size):
                if random.random() < self.p_crossover:
                    new_specie["black_start_board_evaluation"][i][j] = black_start_board_evaluation_2[i][j]
                if random.random() < self.p_crossover:
                    new_specie["black_end_board_evaluation"][i][j] = black_end_board_evaluation_2[i][j]
                if random.random() < self.p_crossover:
                    new_specie["white_start_board_evaluation"][i][j] = white_start_board_evaluation_2[i][j]
                if random.random() < self.p_crossover:
                    new_specie["white_end_board_evaluation"][i][j] = white_end_board_evaluation_2[i][j]

        if random.random() < self.p_crossover:
            new_specie["self_start_action_space_evaluation"] = self_start_action_space_evaluation_2
        if random.random() < self.p_crossover:
            new_specie["self_end_action_space_evaluation"] = self_end_action_space_evaluation_2
        if random.random() < self.p_crossover:
            new_specie["oppo_start_action_space_evaluation"] = oppo_start_action_space_evaluation_2
        if random.random() < self.p_crossover:
            new_specie["oppo_end_action_space_evaluation"] = oppo_end_action_space_evaluation_2
        if random.random() < self.p_crossover:
            new_specie["self_start_unstable_evaluation"] = self_start_unstable_evaluation_2
        if random.random() < self.p_crossover:
            new_specie["self_end_unstable_evaluation"] = self_end_unstable_evaluation_2
        if random.random() < self.p_crossover:
            new_specie["oppo_start_unstable_evaluation"] = oppo_start_unstable_evaluation_2
        if random.random() < self.p_crossover:
            new_specie["oppo_end_unstable_evaluation"] = oppo_end_unstable_evaluation_2
        return new_specie

    def _mutation(self, specie):
        black_start_board_evaluation = specie["black_start_board_evaluation"]
        black_end_board_evaluation = specie["black_end_board_evaluation"]
        white_start_board_evaluation = specie["white_start_board_evaluation"]
        white_end_board_evaluation = specie["white_end_board_evaluation"]
        self_start_action_space_evaluation = specie["self_start_action_space_evaluation"]
        self_end_action_space_evaluation = specie["self_end_action_space_evaluation"]
        oppo_start_action_space_evaluation = specie["oppo_start_action_space_evaluation"]
        oppo_end_action_space_evaluation = specie["oppo_end_action_space_evaluation"]
        self_start_unstable_evaluation = specie["self_start_unstable_evaluation"]
        self_end_unstable_evaluation = specie["self_end_unstable_evaluation"]
        oppo_start_unstable_evaluation = specie["oppo_start_unstable_evaluation"]
        oppo_end_unstable_evaluation = specie["oppo_end_unstable_evaluation"]

        for i in range(self.board_size):
            for j in range(self.board_size):
                black_start_board_evaluation[i][j] = self._mutation_gene(black_start_board_evaluation[i][j])
                black_end_board_evaluation[i][j] = self._mutation_gene(black_end_board_evaluation[i][j])
                white_start_board_evaluation[i][j] = self._mutation_gene(white_start_board_evaluation[i][j])
                white_end_board_evaluation[i][j] = self._mutation_gene(white_end_board_evaluation[i][j])
        self_start_action_space_evaluation = self._mutation_gene(self_start_action_space_evaluation)
        self_end_action_space_evaluation = self._mutation_gene(self_end_action_space_evaluation)
        oppo_start_action_space_evaluation = self._mutation_gene(oppo_start_action_space_evaluation)
        oppo_end_action_space_evaluation = self._mutation_gene(oppo_end_action_space_evaluation)
        self_start_unstable_evaluation = self._mutation_gene(self_start_unstable_evaluation)
        self_end_unstable_evaluation = self._mutation_gene(self_end_unstable_evaluation)
        oppo_start_unstable_evaluation = self._mutation_gene(oppo_start_unstable_evaluation)
        oppo_end_unstable_evaluation = self._mutation_gene(oppo_end_unstable_evaluation)

        specie["black_start_board_evaluation"] = black_start_board_evaluation
        specie["black_end_board_evaluation"] = black_end_board_evaluation
        specie["white_start_board_evaluation"] = white_start_board_evaluation
        specie["white_end_board_evaluation"] = white_end_board_evaluation
        specie["self_start_action_space_evaluation"] = self_start_action_space_evaluation
        specie["self_end_action_space_evaluation"] = self_end_action_space_evaluation
        specie["oppo_start_action_space_evaluation"] = oppo_start_action_space_evaluation
        specie["oppo_end_action_space_evaluation"] = oppo_end_action_space_evaluation
        specie["self_start_unstable_evaluation"] = self_start_unstable_evaluation
        specie["self_end_unstable_evaluation"] = self_end_unstable_evaluation
        specie["oppo_start_unstable_evaluation"] = oppo_start_unstable_evaluation
        specie["oppo_end_unstable_evaluation"] = oppo_end_unstable_evaluation

    def _mutation_gene(self, gene):
        symbol = gene[:3]
        value = gene[3:]
        if random.random() < self.p_mutation:
            symbol = "-0b" if symbol == "+0b" else "+0b"
        for v in value:
            if random.random() < self.p_mutation:
                v = "0" if v == "1" else "1"
            symbol += v
        return symbol

    def train(self, generation):
        self.generate_origin_population()
        for i in range(generation):
            self.selection()
            with open(f"log/{i}_population.log", 'w') as log_file:
                log_file.write(self.trans_population_list.__str__())
            self.cross_over()
            self.mutation()


COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0


class Game(object):
    def __init__(self, board_n):
        self.board_n = board_n
        self.directions = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]

    def get_init_board(self):
        init_board = np.zeros((self.board_n, self.board_n))
        center = self.board_n // 2
        init_board[center - 1, center - 1] = init_board[center, center] = COLOR_WHITE
        init_board[center - 1, center] = init_board[center, center - 1] = COLOR_BLACK
        return init_board

    def move_to_location(self, move):
        return move // self.board_n, move % self.board_n

    def location_to_move(self, location):
        return location[0] * self.board_n + location[1]

    def get_reverse_list(self, board, move, color):
        x, y = move
        reverse_list = []
        if board[x, y] != COLOR_NONE:
            return reverse_list

        for dx, dy in self.directions:
            dir_reverse_list = []
            dir_x, dir_y = x + dx, y + dy
            dir_reverse_flag = False
            while 0 <= dir_x < self.board_n and 0 <= dir_y < self.board_n:
                if board[dir_x, dir_y] == -color:
                    dir_reverse_list.append((dir_x, dir_y))
                    dir_x, dir_y = dir_x + dx, dir_y + dy
                elif board[dir_x, dir_y] == color:
                    dir_reverse_flag = True
                    break
                else:
                    break
            if dir_reverse_flag and len(dir_reverse_list) != 0:
                reverse_list.extend(dir_reverse_list)
        return reverse_list

    def get_next_board(self, board, move, color):
        board = np.copy(board)
        reverse_list = self.get_reverse_list(board, move, color)
        # if len(reverse_list) == 0:
        #     return board, color
        reverse_list.append(move)
        for x, y in reverse_list:
            board[x, y] = color
        if not self.has_legal_move(board, -color):
            return board, color
        else:
            return board, -color

    def get_legal_moves(self, board, color):
        legal_moves = set()
        for i in range(self.board_n):
            for j in range(self.board_n):
                if board[i, j] == color:
                    legal_moves.update(self._get_legal_move_from_location(board, (i, j), color))
        return list(legal_moves)

    def has_legal_move(self, board, color):
        for i in range(self.board_n):
            for j in range(self.board_n):
                if board[i, j] == color and self._check_legal_move_from_location(board, (i, j), color):
                    return True
        return False

    def get_unstable_pieces_num(self, board, view_color, self_legal_moves, oppo_legal_moves):
        self_unstable_pieces = set()
        oppo_unstable_pieces = set()
        for move in oppo_legal_moves:
            self_unstable_pieces.update(self.get_reverse_list(board, move, -view_color))
        for move in self_legal_moves:
            oppo_unstable_pieces.update(self.get_reverse_list(board, move, view_color))
        return len(self_unstable_pieces), len(oppo_unstable_pieces)

    def check_game_end(self, board):
        if len(self.get_legal_moves(board, COLOR_BLACK)) == 0 and len(self.get_legal_moves(board, COLOR_WHITE)) == 0:
            board_sum = np.sum(board)
            if board_sum == 0:
                return True, 0
            elif board_sum > 0:
                return True, -1
            else:
                return True, 1
        else:
            return False, None

    def _get_legal_move_from_location(self, board, location, color):
        x, y = location
        legal_moves = set()
        for dx, dy in self.directions:
            dir_x, dir_y = x + dx, y + dy
            has_reverse_piece = False
            while 0 <= dir_x < self.board_n and 0 <= dir_y < self.board_n:
                if board[dir_x, dir_y] == -color:
                    has_reverse_piece = True
                    dir_x, dir_y = dir_x + dx, dir_y + dy
                elif board[dir_x, dir_y] == color:
                    break
                else:
                    if has_reverse_piece:
                        legal_moves.add((dir_x, dir_y))
                    break
        return legal_moves

    def _check_legal_move_from_location(self, board, location, color):
        x, y = location
        for dx, dy in self.directions:
            dir_x, dir_y = x + dx, y + dy
            has_reverse_piece = False
            while 0 <= dir_x < self.board_n and 0 <= dir_y < self.board_n:
                if board[dir_x, dir_y] == -color:
                    has_reverse_piece = True
                    dir_x, dir_y = dir_x + dx, dir_y + dy
                elif board[dir_x, dir_y] == color:
                    break
                else:
                    if has_reverse_piece:
                        return True
                    break
        return False


MAX_SEARCH = 1
MIN_SEARCH = -1
INF = float('inf')
nINF = float('-inf')


@dataclass
class Node:
    board: np.ndarray
    color: int
    search_type: int
    depth: int
    alpha: float
    beta: float
    value: float


class MinimaxSearchPlayer(object):
    def __init__(self, game, root_board, root_color, search_depth, search_time, start_time, params):
        self.game = game
        self.root_board = root_board
        self.root_color = root_color
        self.search_depth = search_depth
        self.search_time = search_time
        self.root_node = Node(root_board, root_color, MAX_SEARCH, 0, nINF, INF, nINF)
        self.start_time = start_time
        self.root_child_node = []
        self.root_legal_moves = []
        self.black_start_board_evaluation = np.array(params["black_start_board_evaluation"], dtype=float)
        self.black_end_board_evaluation = np.array(params["black_end_board_evaluation"], dtype=float)
        self.white_start_board_evaluation = np.array(params["white_start_board_evaluation"], dtype=float)
        self.white_end_board_evaluation = np.array(params["white_end_board_evaluation"], dtype=float)
        self.self_start_action_space_evaluation = params["self_start_action_space_evaluation"]
        self.self_end_action_space_evaluation = params["self_end_action_space_evaluation"]
        self.oppo_start_action_space_evaluation = params["oppo_start_action_space_evaluation"]
        self.oppo_end_action_space_evaluation = params["oppo_end_action_space_evaluation"]
        self.self_start_unstable_evaluation = params["self_start_unstable_evaluation"]
        self.self_end_unstable_evaluation = params["self_end_unstable_evaluation"]
        self.oppo_start_unstable_evaluation = params["oppo_start_unstable_evaluation"]
        self.oppo_end_unstable_evaluation = params["oppo_end_unstable_evaluation"]

        self.black_board_evaluation_slope = 1.0 * (self.black_end_board_evaluation - self.black_start_board_evaluation) / 60
        self.white_board_evaluation_slope = 1.0 * (self.white_end_board_evaluation - self.white_start_board_evaluation) / 60
        self.self_action_space_evaluation_slope = 1.0 * (self.self_end_action_space_evaluation - self.self_start_action_space_evaluation) / 60
        self.oppo_action_space_evaluation_slope = 1.0 * (self.oppo_end_action_space_evaluation - self.oppo_start_action_space_evaluation) / 60
        self.self_unstable_evaluation_slope = 1.0 * (self.self_end_unstable_evaluation - self.self_start_unstable_evaluation) / 60
        self.oppo_unstable_evaluation_slope = 1.0 * (self.oppo_end_unstable_evaluation - self.oppo_start_unstable_evaluation) / 60

    def minimax_search(self, node):
        if node.depth == 1:
            self.root_child_node.append(node)
        if time.time() - self.start_time > 0.95 * self.search_time:
            node.value = self.get_node_value(node)
            return node.value
        end, winner = self.game.check_game_end(node.board)
        if end:
            if winner == self.root_color:
                node.value = INF
            else:
                node.value = nINF
            return node.value
        if node.depth == self.search_depth:
            node.value = self.get_node_value(node)
            return node.value
        next_moves = self.game.get_legal_moves(node.board, node.color)
        if node.depth == 0:
            self.root_legal_moves = next_moves
        if len(next_moves) == 0:
            next_search_type = -node.search_type
            if next_search_type == MAX_SEARCH:
                next_value = nINF
            else:
                next_value = INF
            next_node = Node(node.board, -node.color, next_search_type, node.depth + 1, node.alpha, node.beta, next_value)
            back_value = self.minimax_search(next_node)
            node.value = back_value

            return node.value
        for move in next_moves:
            next_board, next_color = self.game.get_next_board(node.board, move, node.color)
            next_search_type = -node.search_type if next_color != node.color else node.search_type
            if next_search_type == MAX_SEARCH:
                next_value = nINF
            else:
                next_value = INF
            next_node = Node(next_board, next_color, next_search_type, node.depth + 1, node.alpha, node.beta, next_value)
            back_value = self.minimax_search(next_node)

            if back_value == INF:
                node.value = back_value
                return node.value
            if node.search_type == MAX_SEARCH:
                node.value = max(node.value, back_value)
                if node.value >= node.beta:
                    return node.value
                node.alpha = max(node.alpha, node.value)
            else:
                node.value = min(node.value, back_value)
                if node.value <= node.alpha:
                    return node.value
                node.beta = min(node.beta, node.value)
        return node.value

    def get_node_value(self, node):
        board = node.board
        piece_num = len(np.where(board != COLOR_NONE)[0]) - 4

        if self.root_color == COLOR_BLACK:
            board_evaluation = self.black_start_board_evaluation + piece_num * self.black_board_evaluation_slope
        else:
            board_evaluation = self.white_start_board_evaluation + piece_num * self.white_board_evaluation_slope
        piece_score = np.sum(board_evaluation * board)

        self_action_space = self.game.get_legal_moves(board, self.root_color)
        oppo_action_space = self.game.get_legal_moves(board, -self.root_color)
        self_action_space_coe = self.self_action_space_evaluation_slope * piece_num + self.self_start_action_space_evaluation
        oppo_action_space_coe = self.oppo_action_space_evaluation_slope * piece_num + self.oppo_start_action_space_evaluation
        action_space_score = len(self_action_space) * self_action_space_coe + len(oppo_action_space) * oppo_action_space_coe

        self_unstable_pieces_num, oppo_unstable_pieces_num = self.game.get_unstable_pieces_num(board, self.root_color, self_action_space, oppo_action_space)
        self_unstable_coe = self.self_unstable_evaluation_slope * piece_num + self.self_start_unstable_evaluation
        oppo_unstable_coe = self.oppo_unstable_evaluation_slope * piece_num + self.oppo_start_unstable_evaluation
        unstable_score = self_unstable_coe * self_unstable_pieces_num + oppo_unstable_coe * oppo_unstable_pieces_num

        return piece_score + action_space_score + unstable_score

    def get_action(self):
        max_val = nINF
        best_move = None
        self.minimax_search(self.root_node)
        for i, node in enumerate(self.root_child_node):
            if max_val < node.value:
                max_val = node.value
                best_move = self.root_legal_moves[i]
            if max_val == INF:
                break
        if best_move is None:
            best_move = self.root_legal_moves[0]
        return best_move


if __name__ == '__main__':
    ga = GA(population_size=60,
            gene_len=6,
            board_size=8,
            p_mutation=0.1,
            p_crossover=0.3)
    ga.train(5000)
