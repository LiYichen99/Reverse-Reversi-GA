import numpy as np
# import timeout_decorator

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0

individual = {
    'Vmap_start': [[-17, -43, 25, 23, 23, 25, -43, -17], [-43, -25, 56, 25, 25, 56, -25, -43], [25, 56, 7, -45, -45, 7, 56, 25], [23, 25, -45, -49, -49, -45, 25, 23], [23, 25, -45, -49, -49, -45, 25, 23], [25, 56, 7, -45, -45, 7, 56, 25], [-43, -25, 56, 25, 25, 56, -25, -43], [-17, -43, 25, 23, 23, 25, -43, -17]],
    'Vmap_end': [[46, -40, -5, -9, -9, -5, -40, 46], [-40, -2, 1, -3, -3, 1, -2, -40], [-5, 1, 54, -35, -35, 54, 1, -5], [-9, -3, -35, 21, 21, -35, -3, -9], [-9, -3, -35, 21, 21, -35, -3, -9], [-5, 1, 54, -35, -35, 54, 1, -5], [-40, -2, 1, -3, -3, 1, -2, -40], [46, -40, -5, -9, -9, -5, -40, 46]],
    'mobility_start': [[47, 56, -25, 9, 9, -25, 56, 47], [56, 58, -45, 48, 48, -45, 58, 56], [-25, -45, 36, 33, 33, 36, -45, -25], [9, 48, 33, -45, -45, 33, 48, 9], [9, 48, 33, -45, -45, 33, 48, 9], [-25, -45, 36, 33, 33, 36, -45, -25], [56, 58, -45, 48, 48, -45, 58, 56], [47, 56, -25, 9, 9, -25, 56, 47]],
    'mobility_end': [[34, 41, 43, 56, 56, 43, 41, 34], [41, -52, 36, -24, -24, 36, -52, 41], [43, 36, -9, 5, 5, -9, 36, 43], [56, -24, 5, -36, -36, 5, -24, 56], [56, -24, 5, -36, -36, 5, -24, 56], [43, 36, -9, 5, 5, -9, 36, 43], [41, -52, 36, -24, -24, 36, -52, 41], [34, 41, 43, 56, 56, 43, 41, 34]],
    'stability_weight': 218,
    'frontier_weight_start': -52,
    'frontier_weight_end': -27,
    'diff_weight_start': -52,
    'diff_weight_end': -51
}


class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.game = Game()

    # @timeout_decorator.timeout(4.8)
    def go(self, chessboard):
        try:
            self.candidate_list.clear()
            valids = self.game.get_legal_moves(chessboard, self.color)
            if len(valids) == 0:
                return
            for i in valids:
                self.candidate_list.append(i)
            piece_num = len(np.where(chessboard != 0)[0])
            if piece_num >= 54:
                search_depth = 2
                minimax = MinimaxPlayer(game=self.game, board=chessboard, color=self.color, depth=search_depth,
                                        individual=individual, piece_num=piece_num)
                move = minimax.get_action()
                self.candidate_list.append(move)

                search = SearchPlayer(color=self.color)
                search.search(chessboard, self.color, depth=0)
                move = search.best_move
                if search.best_move is not None:
                    self.candidate_list.append(move)
            else:
                search_depth = 3
                while True:
                    if search_depth >= 20:
                        break
                    minimax = MinimaxPlayer(game=self.game, board=chessboard, color=self.color, depth=search_depth,
                                            individual=individual, piece_num=piece_num)
                    move = minimax.get_action()
                    self.candidate_list.append(move)
                    search_depth += 1
        except:
            print("error")


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
        next_player = self.do_move(b, move, player)
        return b, next_player

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
    def __init__(self, game, board, color, depth, individual, piece_num):
        self.game = game
        self.board = board
        self.root_color = color
        self.depth = depth
        self.root = Node(board, color, 0, 1, nINF, INF, nINF)
        self.root_children = []
        self.root_valids = []
        self.Vmap_start = individual['Vmap_start']
        self.Vmap_end = individual['Vmap_end']
        self.mobility_start = individual['mobility_start']
        self.mobility_end = individual['mobility_end']
        self.stability_weight = individual['stability_weight']
        self.frontier_weight_start = individual['frontier_weight_start']
        self.frontier_weight_end = individual['frontier_weight_end']
        self.diff_weight_start = individual['diff_weight_start']
        self.diff_weight_end = individual['diff_weight_end']
        self.piece_num = piece_num
        self.period = int((self.piece_num - 5) / 30)

    def sigmoid(self, weight):
        if weight == 0:
            weight = 1
        z = 1.0 * (32 - self.piece_num) / weight
        return 1 / (1 + np.exp(-z)) * 1.0

    def getMapWeightSum(self, board, color):
        if self.period == 0:
            return sum(sum(-board * self.Vmap_start * color))
        else:
            return sum(sum(-board * self.Vmap_end * color))

    def get_stability(self, board, color):
        is_stable = np.zeros((8, 8))
        if board[0][0] == color:
            height = 0
            for i in range(8):
                if board[0][i] == color:
                    is_stable[0][i] = 1
                    height = i
                else:
                    break
            for i in range(8):
                if board[i][0] == color:
                    is_stable[i][0] = 1
                    for j in range(height):
                        if board[i][j] == color:
                            is_stable[i][j] = 1
                        else:
                            height = j - 1
                            break
                        height = j
                else:
                    break
            height = 0
            for i in range(8):
                if board[i][0] == color:
                    is_stable[i][0] = 1
                    height = i
                else:
                    break
            for i in range(8):
                if board[0][i] == color:
                    is_stable[0][i] = 1
                    for j in range(height):
                        if board[j][i] == color:
                            is_stable[j][i] = 1
                        else:
                            height = j - 1
                            break
                        height = j
                else:
                    break

        if board[7][0] == color:
            height = 0
            for i in range(8):
                if board[7][i] == color:
                    is_stable[7][i] = 1
                    height = i
                else:
                    break
            for i in range(7, -1, -1):
                if board[i][0] == color:
                    is_stable[i][0] = 1
                    for j in range(height):
                        if board[i][j] == color:
                            is_stable[i][j] = 1
                        else:
                            height = j - 1
                            break
                        height = j
                else:
                    break
            height = 7
            for i in range(7, -1, -1):
                if board[i][0] == color:
                    is_stable[i][0] = 1
                    height = i
                else:
                    break
            for i in range(8):
                if board[7][i] == color:
                    is_stable[7][i] = 1
                    for j in range(7, height, -1):
                        if board[j][i] == color:
                            is_stable[j][i] = 1
                        else:
                            height = j + 1
                            break
                        height = j
                else:
                    break

        if board[7][7] == color:
            height = 7
            for i in range(7, -1, -1):
                if board[7][i] == color:
                    is_stable[7][i] = 1
                    height = i
                else:
                    break
            for i in range(7, -1, -1):
                if board[i][7] == color:
                    is_stable[i][7] = 1
                    for j in range(7, height, -1):
                        if board[i][j] == color:
                            is_stable[i][j] = 1
                        else:
                            height = j + 1
                            break
                        height = j
                else:
                    break
            height = 7
            for i in range(7, -1, -1):
                if board[i][7] == color:
                    is_stable[i][7] = 1
                    height = i
                else:
                    break
            for i in range(7, -1, -1):
                if board[7][i] == color:
                    is_stable[7][i] = 1
                    for j in range(7, height, -1):
                        if board[j][i] == color:
                            is_stable[j][i] = 1
                        else:
                            height = j + 1
                            break
                        height = j
                else:
                    break

        if board[0][7] == color:
            height = 7
            for i in range(7, -1, -1):
                if board[0][i] == color:
                    is_stable[0][i] = 1
                    height = i
                else:
                    break
            for i in range(8):
                if board[i][7] == color:
                    is_stable[i][7] = 1
                    for j in range(7, height, -1):
                        if board[i][j] == color:
                            is_stable[i][j] = 1
                        else:
                            height = j + 1
                            break
                        height = j
                else:
                    break
            height = 0
            for i in range(8):
                if board[i][7] == color:
                    is_stable[i][7] = 1
                    height = i
                else:
                    break
            for i in range(7, -1, -1):
                if board[0][i] == color:
                    is_stable[0][i] = 1
                    for j in range(height):
                        if board[j][i] == color:
                            is_stable[j][i] = 1
                        else:
                            height = j - 1
                            break
                        height = j
                else:
                    break

        return len(np.where(is_stable == 1)[0])

    def get_mobility(self, board, color):
        mobility = 0
        if self.period == 0:
            for i, j in self.game.get_legal_moves(board, color):
                mobility += self.mobility_start[i][j]
            for i, j in self.game.get_legal_moves(board, -color):
                mobility -= self.mobility_start[i][j]
            return mobility
        else:
            for x, y in self.game.get_legal_moves(board, color):
                mobility += self.mobility_end[x][y]
            for x, y in self.game.get_legal_moves(board, -color):
                mobility -= self.mobility_end[x][y]
            return mobility

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
                if board[i][j] == color:
                    for x_direction, y_direction in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0],
                                                     [-1, 1]]:
                        x = i + x_direction
                        y = j + y_direction
                        if 0 <= x < 8 and 0 <= y < 8 and board[x][y] == 0:
                            count += 1
                            break
        return count

    def get_diff(self, board, color):
        return len(np.where(board == color)[0]) - len(np.where(board == -color)[0]) * (
            self.diff_weight_start if self.period == 0 else self.diff_weight_end)

    def getValue(self, node):
        board = node.board
        mapWeightSum = self.getMapWeightSum(board, self.root_color)
        stability = (self.get_stability(board, -self.root_color) - self.get_stability(board,
                                                                                      self.root_color)) * self.stability_weight
        mobility = self.get_mobility(board, self.root_color)
        frontier = (self.get_frontier(board, self.root_color) - self.get_frontier(board, -self.root_color)) * (
            self.frontier_weight_start if self.period == 0 else self.frontier_weight_end)
        diff = self.get_diff(board, self.root_color)
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
        if node.depth == self.depth:
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
            subNode = Node(node.board, -node.color, node.depth + 1, -node.search_type, node.alpha, node.beta,
                           next_value)
            if node.depth == 0:
                self.root_children.append(subNode)
            node.value = self.minimax(subNode)
            return node.value
        for move in valids:
            next_board, _ = self.game.getNextState(node.board, node.color, move)
            if node.search_type == 1:
                next_value = INF
            else:
                next_value = nINF
            subNode = Node(next_board, -node.color, node.depth + 1, -node.search_type, node.alpha, node.beta,
                           next_value)
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


class SearchPlayer(object):
    def __init__(self, color):
        self.game = Game()
        self.root_color = color
        self.best_move = None

    def search(self, board, color, depth):
        end, winner = self.game.is_game_end(board)
        if end:
            if winner == self.root_color:
                return 1
            elif winner == 0:
                return 0
            else:
                return -1
        valids = self.game.get_legal_moves(board, color)
        return_values = []
        for move in valids:
            next_board, next_color = self.game.getNextState(board, color, move)
            value = self.search(next_board, next_color, depth + 1)
            return_values.append(value)
            if color == self.root_color:
                if value == 1:
                    if depth == 0:
                        self.best_move = move
                    return 1
            else:
                if value == -1:
                    return -1

        for i in range(len(return_values)):
            if return_values[i] == 0:
                if color == self.root_color and depth == 0:
                    self.best_move = valids[i]
                return 0
        if color == self.root_color:
            return -1
        else:
            return 1

if __name__ == '__main__':
    l = [[34, 41, 43, 56, 56, 43, 41, 34], [41, -52, 36, -24, -24, 36, -52, 41], [43, 36, -9, 5, 5, -9, 36, 43], [56, -24, 5, -36, -36, 5, -24, 56], [56, -24, 5, -36, -36, 5, -24, 56], [43, 36, -9, 5, 5, -9, 36, 43], [41, -52, 36, -24, -24, 36, -52, 41], [34, 41, 43, 56, 56, 43, 41, 34]]
    for i in range(8):
        for j in range(8):
            print(l[i][j], end="")
            if j < 7:
                print("& ", end="")
            else:
                print("\\\\", end="")
        print()