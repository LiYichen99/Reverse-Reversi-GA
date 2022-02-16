import multiprocessing
import random
import time

import numpy as np


class GA(object):
    def __init__(self, size, gene_length, selection_size, p_mutation, train_times, chrome_length=16):
        self.size = size
        self.gene_length = gene_length
        self.selection_size = selection_size
        self.p_mutation = p_mutation
        self.population = []
        self.chrome_length = chrome_length
        self.train_times = train_times

    def do_compete(self, idv_index):
        game = Game()
        win = 0
        lose = 0
        draw = 0
        for i in range(self.size):
            if i != idv_index:
                # start_time = time.time()
                if random.randint(0, 1) == 1:
                    color = 1
                else:
                    color = -1
                board = game.get_init_board()
                end, winner = game.is_game_end(board)
                current_color = -1
                # game.display(board)
                while not end:
                    # print(b.get_legal_moves(current_color))
                    if current_color == color:
                        player = MinimaxPlayer(game, board, color, 5, 1, self.population[idv_index])
                    else:
                        player = MinimaxPlayer(game, board, -color, 5, 1, self.population[i])
                    current_color = game.do_move(board, player.get_action(), current_color)
                    end, winner = game.is_game_end(board)
                    # game.display(board)
                if winner == color:
                    win += 1
                elif winner == -color:
                    lose += 1
                else:
                    draw += 1
                # game.display(board)
            # print(i, time.time() - start_time)
        return win, lose, draw

    def compete(self):
        start_time = time.time()
        with multiprocessing.Pool(multiprocessing.cpu_count()-1) as p:
            result = p.map(self.do_compete, range(self.size))
        p.close()
        p.join()
        for i in range(self.size):
            self.population[i]['win'] = result[i][0]
            self.population[i]['lose'] = result[i][1]
            self.population[i]['draw'] = result[i][2]
        print(time.time() - start_time)

    def selection(self):
        self.compete()
        self.population = sorted(self.population, key=lambda individual: individual['win'], reverse=True)
        self.population = self.population[:self.selection_size]
        for individual in self.population:
            individual['generation'] += 1

    def crossover(self):
        for i in range(self.selection_size):
            j = np.random.choice(range(self.selection_size))
            while i == j:
                j = np.random.choice(range(self.selection_size))
            new_idv = self.do_crossover(self.population[i], self.population[j])
            self.population.append(new_idv)

    def do_crossover(self, individual1, individual2):
        new_idv = dict()
        chrome = []
        for i in range(self.chrome_length):
            x = random.randint(0, 1)
            if x == 0:
                chrome.append(self.mutation(individual1['chrome'][i]))
            else:
                chrome.append(self.mutation(individual2['chrome'][i]))
        new_idv['chrome'] = chrome
        new_idv['generation'] = 1
        self.transform(new_idv)
        return new_idv

    def mutation(self, gene):
        new_gene = []
        for i in range(self.gene_length):
            if random.random() < self.p_mutation:
                new_gene.append(1) if gene[i] == 0 else new_gene.append(0)
            else:
                new_gene.append(gene[i])
        return new_gene

    def transform(self, individual):
        chrome = individual['chrome']
        Vmap = np.zeros((8, 8))
        Vmap[0][0] = Vmap[7][0] = Vmap[0][7] = Vmap[7][7] = self.transform_gene(chrome[0])
        Vmap[1][0] = Vmap[0][1] = Vmap[6][0] = Vmap[0][6] = Vmap[7][1] = Vmap[1][7] = Vmap[6][7] = Vmap[7][6] = self.transform_gene(chrome[1])
        Vmap[1][1] = Vmap[1][6] = Vmap[6][1] = Vmap[6][6] = self.transform_gene(chrome[2])
        Vmap[0][2] = Vmap[0][5] = Vmap[2][0] = Vmap[2][7] = Vmap[5][0] = Vmap[5][7] = Vmap[7][2] = Vmap[7][5] = self.transform_gene(chrome[3])
        Vmap[1][2] = Vmap[1][5] = Vmap[2][1] = Vmap[2][6] = Vmap[5][1] = Vmap[5][6] = Vmap[6][2] = Vmap[6][5] = self.transform_gene(chrome[4])
        Vmap[2][2] = Vmap[2][5] = Vmap[5][2] = Vmap[5][5] = self.transform_gene(chrome[5])
        Vmap[0][3] = Vmap[0][4] = Vmap[3][0] = Vmap[3][7] = Vmap[4][0] = Vmap[4][7] = Vmap[7][3] = Vmap[7][4] = self.transform_gene(chrome[6])
        Vmap[1][3] = Vmap[1][4] = Vmap[3][1] = Vmap[3][6] = Vmap[4][1] = Vmap[4][6] = Vmap[6][3] = Vmap[6][4] = self.transform_gene(chrome[7])
        Vmap[2][3] = Vmap[2][4] = Vmap[3][2] = Vmap[3][5] = Vmap[4][2] = Vmap[4][5] = Vmap[5][3] = Vmap[5][4] = self.transform_gene(chrome[8])
        Vmap[3][3] = Vmap[3][4] = Vmap[4][3] = Vmap[4][4] = self.transform_gene(chrome[9])
        individual['Vmap'] = Vmap
        individual['Vmap_weight'] = self.transform_gene(chrome[10])
        individual['stability_weight'] = self.transform_gene(chrome[11])
        individual['mobility_weight'] = self.transform_gene(chrome[12])
        individual['parity_weight'] = self.transform_gene(chrome[13])
        individual['frontier_weight'] = self.transform_gene(chrome[14])
        individual['diff_weight'] = self.transform_gene(chrome[15])

    def transform_gene(self, gene):
        i = self.gene_length - 1
        result = 0
        a = 1
        while i > 0:
            result = result + a * gene[i]
            a *= 2
            i -= 1
        if gene[0] == 1:
            result = -result
        return result

    def genetic_algorithm(self):
        self.population = []
        for i in range(self.size):
            individual = dict()
            chrome = []
            for j in range(self.chrome_length):
                gene = []
                for k in range(self.gene_length):
                    gene.append(random.randint(0, 1))
                chrome.append(gene)
            individual['chrome'] = chrome
            individual['generation'] = 1
            self.transform(individual)
            self.population.append(individual)
        for i in range(self.train_times):
            start_time = time.time()
            self.selection()
            with open(f"log/{i}_population.log", 'w') as log_file:
                log_file.write(self.population.__str__())
                log_file.write("\n")
            self.crossover()
            print(time.time() - start_time)


class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.game = Game()

    def go(self, chessboard):
        self.candidate_list.clear()
        valids = self.game.get_legal_moves(chessboard, self.color)
        for i in valids:
            self.candidate_list.append(i)


class Game(object):
    chess = {-1: "X", 0: ".", 1: "O"}

    def get_init_board(self):
        board = np.zeros((8, 8))
        board[3][3] = 1
        board[3][4] = -1
        board[4][3] = -1
        board[4][4] = 1
        return board

    @staticmethod
    def is_on_board(x, y):
        return 0 <= x < 8 and 0 <= y < 8

    def is_legal_move(self, board, color, move):
        x_start, y_start = move
        if not (self.is_on_board(x_start, y_start) and board[x_start][y_start] == 0):
            return False
        for x_direction, y_direction in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
            x = x_start
            y = y_start
            x += x_direction
            y += y_direction
            if not self.is_on_board(x, y) or board[x][y] == 0 or board[x][y] == color:
                continue
            x += x_direction
            y += y_direction
            while self.is_on_board(x, y):
                if board[x][y] == 0:
                    break
                elif board[x][y] == color:
                    return True
                x += x_direction
                y += y_direction
        return False

    def get_legal_moves(self, board, color):
        moves = []
        for i in range(8):
            for j in range(8):
                if self.is_legal_move(board, color, (i, j)):
                    moves.append((i, j))
        return moves

    def get_reverse_list(self, board, move, color):
        x_start, y_start = move
        reverse_set = set()
        for x_direction, y_direction in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
            x = x_start
            y = y_start
            x += x_direction
            y += y_direction
            if not self.is_on_board(x, y) or board[x][y] != -color:
                continue
            reverse_temp = [(x, y)]
            x += x_direction
            y += y_direction
            while self.is_on_board(x, y):
                if board[x][y] == 0:
                    break
                elif board[x][y] == -color:
                    reverse_temp.append((x, y))
                    x += x_direction
                    y += y_direction
                else:
                    reverse_set.update(reverse_temp)
                    break
        return list(reverse_set)

    def do_move(self, board, move, color):
        reverse_list = self.get_reverse_list(board, move, color)
        assert len(reverse_list) > 0
        reverse_list.append(move)
        for i, j in reverse_list:
            board[i][j] = color
        if len(self.get_legal_moves(board, -color)) == 0:
            return color
        else:
            return -color

    def display(self, board):
        print(' ', '0 1 2 3 4 5 6 7')
        for i in range(8):
            print(i, end=' ')
            for j in range(8):
                print(self.chess[board[i][j]], end=' ')
            print()

    def getNextState(self, board, player, move):
        b = np.copy(board)
        self.do_move(b, move, player)
        return b

    def is_game_end(self, board):
        if len(self.get_legal_moves(board, 1)) != 0 or len(self.get_legal_moves(board, -1)) != 0:
            return 0, 0
        count = np.sum(board)
        if count > 0:
            return 1, -1
        elif count < 0:
            return 1, 1
        else:
            return 1, 0


INF = float('inf')
nINF = float('-inf')


class Node(object):
    def __init__(self, board, color, depth, search_type, alpha, beta, value):
        self.board = board
        self.color = color
        self.depth = depth
        self.search_type = search_type
        self.alpha = alpha
        self.beta = beta
        self.value = value


class UnionNode:

    def __init__(self):
        self.fa = self
        self.nums = 1

    @staticmethod
    def find(n):
        if n.fa == n:
            return n
        else:
            n.fa = UnionNode.find(n.fa)
            return n.fa

    @staticmethod
    def union(a, b):
        a_fa, b_fa = UnionNode.find(a), UnionNode.find(b)
        if a_fa == b_fa:
            return
        if a_fa.nums > b_fa.nums:
            b_fa.fa = a_fa
            a_fa.nums += b_fa.nums
        else:
            a_fa.fa = b.fa
            b_fa.nums += a_fa.nums


class MinimaxPlayer(object):
    def __init__(self, game, board, color, time_out, depth, individual):
        self.game = game
        self.board = board
        self.root_color = color
        self.time_out = time_out
        self.depth = depth
        self.start_time = time.time()
        self.root = Node(board, color, 0, 1, nINF, INF, nINF)
        self.root_children = []
        self.root_valids = []
        self.Vmap = individual['Vmap']
        self.Vmap_weight = individual['Vmap_weight']
        self.stability_weight = individual['stability_weight']
        self.mobility_weight = individual['mobility_weight']
        self.parity_weight = individual['parity_weight']
        self.frontier_weight = individual['frontier_weight']
        self.diff_weight = individual['diff_weight']
        self.piece_num = 4

    def sigmoid(self, weight):
        if weight == 0:
            weight = 1
        z = 1.0 * (32 - self.piece_num) / weight
        return 1 / (1 + np.exp(-z)) * 1.0

    def getMapWeightSum(self, board, color):
        return sum(sum(board * self.Vmap)) * color

    def get_stability(self, board, color):
        valids = self.game.get_legal_moves(board, -color)
        reverse_set = set()
        for move in valids:
            reverse_list = self.game.get_reverse_list(board, move, -color)
            reverse_set.update(reverse_list)
        return len(np.where(board == color)[0]) - len(reverse_set)

    def get_mobility(self, board, color):
        return len(self.game.get_legal_moves(board, color))

    def get_parity(self, board):
        space_list = np.where(board == 0)
        space_list = [(space_list[0][i], space_list[1][i]) for i in range(len(space_list[0]))]

        space_node_dict = {}
        for space in space_list:
            space_node_dict[space] = UnionNode()
        for x, y in space_list:
            if (x - 1, y) in space_list:
                UnionNode.union(space_node_dict[(x, y)], space_node_dict[(x - 1, y)])
            if (x, y - 1) in space_list:
                UnionNode.union(space_node_dict[(x, y)], space_node_dict[(x, y - 1)])
        even_area = set()
        for space in space_list:
            n = space_node_dict[space]
            if n.nums % 2 == 0:
                even_area.add(n)
        return len(even_area)

    def get_frontier(self, board, color):
        count = 0
        for i in range(8):
            for j in range(8):
                if board[i][j] == -color:
                    for x_direction, y_direction in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0],
                                                     [-1, 1]]:
                        x = i + x_direction
                        y = j + y_direction
                        if 0 <= x < 8 and 0 <= y < 8 and board[x][y] == 0:
                            count += 1
                            break
        return count

    def get_diff(self, board, color):
        return len(np.where(board == -color)[0]) - len(np.where(board == color)[0])

    def getValue(self, node):
        board = node.board
        color = self.root_color
        self.piece_num = len(np.where(board != 0)[0])
        mapWeightSum = self.getMapWeightSum(board, color) * self.sigmoid(self.Vmap_weight)
        stability = (self.get_stability(board, -color) - self.get_stability(board, color)) * self.sigmoid(
            self.stability_weight)
        mobility = (self.get_mobility(board, color) - self.get_mobility(board, -color)) * self.sigmoid(
            self.mobility_weight)
        parity = self.get_parity(board) * self.sigmoid(self.parity_weight)
        frontier = (self.get_frontier(board, color) - self.get_frontier(board, -color)) * self.sigmoid(
            self.frontier_weight)
        diff = self.get_diff(board, color) * self.sigmoid(self.diff_weight)
        return mapWeightSum + stability + mobility + parity + frontier + diff

    def get_action(self):
        self.minimax(self.root)
        max_value = nINF
        best_move = self.root_valids[0]
        for i, child in enumerate(self.root_children):
            if child.value > max_value:
                max_value = child.value
                best_move = self.root_valids[i]
            if child.value == INF:
                return best_move
        return best_move

    def minimax(self, node):
        end, winner = self.game.is_game_end(node.board)
        if end:
            node.value = INF * winner * self.root_color
            return node.value
        if time.time() - self.start_time > self.time_out or node.depth == self.depth:
            node.value = self.getValue(node)
            return node.value
        valids = self.game.get_legal_moves(node.board, node.color)
        if node.depth == 0:
            self.root_valids = valids
        if len(valids) == 0:
            if node.search_type == 1:
                next_value = INF
            else:
                next_value = nINF
            subNode = Node(node.board, -node.color, node.depth + 1, -node.search_type, node.alpha, node.beta, next_value)
            if node.depth == 0:
                self.root_children.append(subNode)
            node.value = self.minimax(subNode)
            return node.value
        for move in valids:
            next_board = self.game.getNextState(node.board, node.color, move)
            if node.search_type == 1:
                next_value = INF
            else:
                next_value = nINF
            subNode = Node(next_board, -node.color, node.depth + 1, -node.search_type, node.alpha, node.beta, next_value)
            if node.depth == 0:
                self.root_children.append(subNode)
            utility = self.minimax(subNode)
            if utility == INF:
                node.value = INF
                return INF
            if node.search_type == -1:
                node.value = min(node.value, utility)
                if node.value <= node.alpha:
                    return node.value
                node.beta = min(node.value, node.beta)
            else:
                node.value = max(node.value, utility)
                if node.value >= node.beta:
                    return node.value
                node.alpha = max(node.value, node.alpha)
        return node.value


if __name__ == '__main__':
    ga = GA(size=60, gene_length=7, selection_size=30, p_mutation=0.05, train_times=5000)
    ga.genetic_algorithm()
