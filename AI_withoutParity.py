import multiprocessing
import random
import time

import numpy as np


class GA(object):
    def __init__(self, size, gene_length, selection_size, p_mutation, train_times, chrome_length=15):
        self.size = size
        self.gene_length = gene_length
        self.selection_size = selection_size
        self.p_mutation = p_mutation
        self.population = []
        self.chrome_length = chrome_length
        self.train_times = train_times

    def do_compete(self, idv_index):
        game = Game()
        b = Board()
        win = 0
        lose = 0
        draw = 0
        for i in range(self.size):
            if i != idv_index:
                # start_time = time.time()
                for color in (-1, 1):
                    board = b.get_init_board()
                    end, winner = game.is_game_end(board)
                    current_color = -1
                    # game.display(board)
                    while not end:
                        # print(b.get_legal_moves(current_color))
                        player1 = MinimaxPlayer(game, board, color, 5, 1, self.population[idv_index])
                        player2 = MinimaxPlayer(game, board, -color, 5, 1, self.population[i])
                        players = {color: player1, -color: player2}
                        player = players[current_color]
                        current_color = b.do_move(player.get_action(), current_color)
                        end, winner = game.is_game_end(board)
                        board = b.board
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
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
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
        select_population = self.population[:self.selection_size]
        for individual in select_population:
            individual['fitness'] += 1
        return select_population

    def crossover(self, select_population):
        random.shuffle(self.population)
        for i in range(self.size - self.selection_size):
            j = np.random.choice(range(self.size - self.selection_size))
            while i == j:
                j = np.random.choice(range(self.size - self.selection_size))
            new_idv = self.do_crossover(self.population[i], self.population[j])
            select_population.append(new_idv)
        self.population = select_population

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
        new_idv['fitness'] = 1
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
        map = np.zeros((4, 4))
        c = 0
        chrome = individual['chrome']
        for i in range(4):
            for j in range(i+1):
                map[i][j] = self.transform_gene(chrome[c])
                c += 1
        map += map.T - np.diag(map.diagonal())
        map2 = np.rot90(map, 1)
        Vmap1 = np.append(map, map2, axis=0)
        map2 = np.rot90(map2, 1)
        map = np.rot90(map, -1)
        Vmap2 = np.append(map, map2, axis=0)
        Vmap = np.append(Vmap1, Vmap2, axis=1)
        individual['Vmap'] = Vmap
        individual['Vmap_weight'] = self.transform_gene(chrome[10])
        individual['stability_weight'] = self.transform_gene(chrome[11])
        individual['mobility_weight'] = self.transform_gene(chrome[12])
        # individual['parity_weight'] = self.transform_gene(chrome[13])
        individual['frontier_weight'] = self.transform_gene(chrome[13])
        individual['diff_weight'] = self.transform_gene(chrome[14])

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
            individual['fitness'] = 1
            self.transform(individual)
            self.population.append(individual)
        for i in range(self.train_times):
            start_time = time.time()
            select_population = self.selection()
            with open(f"log/{i}_population.log", 'w') as log_file:
                log_file.write(select_population.__str__())
            self.crossover(select_population)
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
        valids = self.game.getValidMoves(chessboard, self.color)
        for i in valids:
            self.candidate_list.append(i)


class Board(object):
    def __init__(self):
        self.board = np.zeros((8, 8))
        self.board[3][3] = 1
        self.board[3][4] = -1
        self.board[4][3] = -1
        self.board[4][4] = 1

    def get_init_board(self):
        self.__init__()
        return self.board

    @staticmethod
    def is_on_board(x, y):
        return 0 <= x < 8 and 0 <= y < 8

    def is_legal_move(self, color, move):
        x_start, y_start = move
        if not (self.is_on_board(x_start, y_start) and self.board[x_start][y_start] == 0):
            return False
        for x_direction, y_direction in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
            x = x_start
            y = y_start
            x += x_direction
            y += y_direction
            if not self.is_on_board(x, y) or self.board[x][y] == 0 or self.board[x][y] == color:
                continue
            x += x_direction
            y += y_direction
            while self.is_on_board(x, y):
                if self.board[x][y] == 0:
                    break
                elif self.board[x][y] == color:
                    return True
                x += x_direction
                y += y_direction
        return False

    def get_legal_moves(self, color):
        moves = []
        for i in range(8):
            for j in range(8):
                if self.is_legal_move(color, (i, j)):
                    moves.append((i, j))
        return moves

    def get_reverse_list(self, move, color):
        x_start, y_start = move
        reverse_set = set()
        for x_direction, y_direction in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
            x = x_start
            y = y_start
            x += x_direction
            y += y_direction
            if not self.is_on_board(x, y) or self.board[x][y] != -color:
                continue
            reverse_temp = [(x, y)]
            x += x_direction
            y += y_direction
            while self.is_on_board(x, y):
                if self.board[x][y] == 0:
                    break
                elif self.board[x][y] == -color:
                    reverse_temp.append((x, y))
                    x += x_direction
                    y += y_direction
                else:
                    reverse_set.update(reverse_temp)
                    break
        return list(reverse_set)

    def do_move(self, move, color):
        reverse_list = self.get_reverse_list(move, color)
        assert len(reverse_list) > 0
        reverse_list.append(move)
        for i, j in reverse_list:
            self.board[i][j] = color
        if len(self.get_legal_moves(-color)) == 0:
            return color
        else:
            return -color


class Game(object):
    chess = {-1: "X", 0: ".", 1: "O"}

    def display(self, board):
        print(' ', '0 1 2 3 4 5 6 7')
        for i in range(8):
            print(i, end=' ')
            for j in range(8):
                print(self.chess[board[i][j]], end=' ')
            print()

    def getNextState(self, board, player, move):
        b = Board()
        b.board = np.copy(board)
        b.do_move(move, player)
        return b.board

    def getValidMoves(self, board, player):
        b = Board()
        b.board = np.copy(board)
        return b.get_legal_moves(player)

    def is_game_end(self, board):
        if len(self.getValidMoves(board, 1)) != 0 or len(self.getValidMoves(board, -1)) != 0:
            return 0, 0
        count = np.sum(board)
        if count > 0:
            return 1, 1
        elif count < 0:
            return 1, -1
        else:
            return 1, 0


INF = float('inf')
nINF = float('-inf')


class Node(object):
    def __init__(self, board, color, depth, type, alpha, beta, value):
        self.board = board
        self.color = color
        self.depth = depth
        self.type = type
        self.alpha = alpha
        self.beta = beta
        self.value = value


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
        # self.parity_weight = individual['parity_weight']
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
        b = Board()
        b.board = np.copy(board)
        valids = b.get_legal_moves(-color)
        reverse_set = set()
        for move in valids:
            reverse_list = b.get_reverse_list(move, -color)
            reverse_set.update(reverse_list)
        return len(np.where(board == color)[0]) - len(reverse_set)

    def get_mobility(self, board, color):
        b = Board()
        b.board = np.copy(board)
        return len(b.get_legal_moves(color))

    def get_parity(self, board):
        b = np.copy(board)
        is_visited = np.zeros((8, 8))
        even_count = 0
        for x in range(8):
            for y in range(8):
                if b[x][y] == 0 and is_visited[x][y] == 0:
                    is_visited[x][y] = 1
                    count = self.bfs(b, is_visited, x, y, count=1)
                    if count % 2 == 0:
                        even_count += 1
        return even_count

    def bfs(self, board, is_visited, x, y, count):
        for x_direction, y_direction in [[0, 1], [1, 0], [0, -1], [-1, 0]]:
            x += x_direction
            y += y_direction
            if 0 <= x < 8 and 0 <= y < 8 and board[x][y] == 0 and is_visited[x][y] == 0:
                is_visited[x][y] = 1
                count += 1
                count = self.bfs(board, is_visited, x, y, count)
        return count

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
        # parity = self.get_parity(board) * self.sigmoid(self.parity_weight)
        frontier = (self.get_frontier(board, color) - self.get_frontier(board, -color)) * self.sigmoid(
            self.frontier_weight)
        diff = self.get_diff(board, color) * self.sigmoid(self.diff_weight)
        return mapWeightSum + stability + mobility + frontier + diff

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
        valids = self.game.getValidMoves(node.board, node.color)
        if node.depth == 0:
            self.root_valids = valids
        if len(valids) == 0:
            if node.type == 1:
                next_value = INF
            else:
                next_value = nINF
            subNode = Node(node.board, -node.color, node.depth + 1, -node.type, node.alpha, node.beta, next_value)
            if node.depth == 0:
                self.root_children.append(subNode)
            node.value = self.minimax(subNode)
            return node.value
        for move in valids:
            next_board = self.game.getNextState(node.board, node.color, move)
            if node.type == 1:
                next_value = INF
            else:
                next_value = nINF
            subNode = Node(next_board, -node.color, node.depth + 1, -node.type, node.alpha, node.beta, next_value)
            if node.depth == 0:
                self.root_children.append(subNode)
            utility = self.minimax(subNode)
            if utility == INF:
                node.value = INF
                return INF
            if node.type == -1:
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
