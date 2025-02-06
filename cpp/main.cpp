#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>
#include <memory>
#include <map>

// Define the Tic-Tac-Toe game
class TicTacToe {
public:
    TicTacToe() : board(3, std::vector<int>(3, 0)), current_player(1) {}

    void reset() {
        board = std::vector<std::vector<int>>(3, std::vector<int>(3, 0));
        current_player = 1;
    }

    bool is_draw() const {
        for (const auto& row : board) {
            for (int cell : row) {
                if (cell == 0) return false;
            }
        }
        return true;
    }

    bool is_win(int player) const {
        for (int i = 0; i < 3; ++i) {
            if (board[i][0] == player && board[i][1] == player && board[i][2] == player) return true;
            if (board[0][i] == player && board[1][i] == player && board[2][i] == player) return true;
        }
        if (board[0][0] == player && board[1][1] == player && board[2][2] == player) return true;
        if (board[0][2] == player && board[1][1] == player && board[2][0] == player) return true;
        return false;
    }

    bool is_game_over() const {
        return is_win(1) || is_win(-1) || is_draw();
    }

    bool make_move(int row, int col) {
        if (board[row][col] == 0) {
            board[row][col] = current_player;
            current_player = -current_player;
            return true;
        }
        return false;
    }

    void print_board() const {
        for (const auto& row : board) {
            for (int cell : row) {
                if (cell == 1) std::cout << "X ";
                else if (cell == -1) std::cout << "O ";
                else std::cout << ". ";
            }
            std::cout << "\n";
        }
    }

    std::vector<std::vector<int>> get_board() const {
        return board;
    }

    int get_current_player() const {
        return current_player;
    }

private:
    std::vector<std::vector<int>> board;
    int current_player;
};

// Define a simple FeedForward neural network
struct Net : torch::nn::Module {
    torch::nn::Linear fc1, fc2, fc3;

    Net(int64_t input_size, int64_t hidden_size, int64_t output_size)
        : fc1(input_size, hidden_size), fc2(hidden_size, hidden_size), fc3(hidden_size, output_size) {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        x = fc3(x);
        return x;
    }
};

// Monte Carlo Tree Search Node
class MCTSNode : public std::enable_shared_from_this<MCTSNode> {
public:
    MCTSNode(std::shared_ptr<MCTSNode> parent, TicTacToe state)
        : parent(parent), state(state), visits(0), wins(0) {}

    std::shared_ptr<MCTSNode> select() {
        std::shared_ptr<MCTSNode> best_child = nullptr;
        double best_value = -std::numeric_limits<double>::infinity();
        for (auto& child : children) {
            double uct_value = child->wins / (child->visits + 1e-6) +
                               std::sqrt(2 * std::log(visits + 1) / (child->visits + 1e-6));
            if (uct_value > best_value) {
                best_value = uct_value;
                best_child = child;
            }
        }
        return best_child;
    }

    void expand() {
        // Only expand if the current state is not terminal (game over)
        if (state.is_game_over()) {
            return;
        }

        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                if (state.get_board()[row][col] == 0) {
                    TicTacToe new_state = state;
                    new_state.make_move(row, col);
                    children.push_back(std::make_shared<MCTSNode>(shared_from_this(), new_state));
                }
            }
        }
    }

    void backpropagate(int result) {
        visits++;
        wins += result;
        if (parent) {
            parent->backpropagate(-result);
        }
    }

    std::shared_ptr<MCTSNode> best_child() {
        std::shared_ptr<MCTSNode> best_child = nullptr;
        double best_value = -std::numeric_limits<double>::infinity();
        for (auto& child : children) {
            double value = child->wins / (child->visits + 1e-6);
            if (value > best_value) {
                best_value = value;
                best_child = child;
            }
        }
        return best_child;
    }

    TicTacToe state;
    std::shared_ptr<MCTSNode> parent;
    std::vector<std::shared_ptr<MCTSNode>> children;
    int visits;
    double wins;
};

// Monte Carlo Tree Search
class MCTS {
public:
    MCTS(TicTacToe initial_state) : root(std::make_shared<MCTSNode>(nullptr, initial_state)) {}

    void run(int iterations) {
        for (int i = 0; i < iterations; ++i) {
            std::shared_ptr<MCTSNode> node = root;
            while (!node->children.empty()) {
                node = node->select();
            }
            if (!node->state.is_game_over()) {
                node->expand();
                node = node->select();
            }
            int result = simulate(node->state);
            node->backpropagate(result);
        }
    }

    TicTacToe best_move() {
        return root->best_child()->state;
    }

private:
    int simulate(TicTacToe state) {
        while (!state.is_game_over()) {
            std::vector<std::pair<int, int>> available_moves;
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    if (state.get_board()[row][col] == 0) {
                        available_moves.emplace_back(row, col);
                    }
                }
            }

            // If no available moves are left, the game should end as a draw
            if (available_moves.empty()) {
                return 0; // Draw
            }

            // Select a random available move
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, available_moves.size() - 1);
            auto move = available_moves[dis(gen)];
            state.make_move(move.first, move.second);
        }

        if (state.is_win(1)) return 1;  // Player X wins
        if (state.is_win(-1)) return -1; // Player O wins
        return 0; // Draw
    }

    std::shared_ptr<MCTSNode> root;
};

int main() {
    TicTacToe game;
    Net net(9, 64, 9);
    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));

    for (int episode = 0; episode < 10; ++episode) { // Reduced the number of episodes for demonstration
        std::cout << "Starting episode " << episode + 1 << "...\n";
        game.reset();

        while (!game.is_game_over()) {
            MCTS mcts(game);
            mcts.run(1000);
            game = mcts.best_move();
            game.print_board();
            std::cout << "\n";
        }

        // Determine the winner
        if (game.is_win(1)) {
            std::cout << "Player X (1) wins!\n";
        } else if (game.is_win(-1)) {
            std::cout << "Player O (-1) wins!\n";
        } else {
            std::cout << "It's a draw!\n";
        }
        std::cout << "========================\n";
    }

    return 0;
}
