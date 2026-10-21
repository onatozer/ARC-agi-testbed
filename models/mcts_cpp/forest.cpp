// A forest of PUCT search trees (one per puzzle) with the baseline MCTS's exact arithmetic, kept in C++ so that a
// round's selections and backups cost microseconds instead of a Python walk per tree. Nodes are ids into one array;
// the domain state of node i is kept by the Python side at the same index, so this file never sees a latent.
// Built on demand by models/mcts_cpp/__init__.py (torch.utils.cpp_extension.load).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <limits>
#include <vector>

namespace py = pybind11;

struct Node {
    int tree;
    int parent;
    int depth_left;
    double prior;
    int first_child = -1;
    int num_children = 0;
    int visits = 0;
    int unobserved = 0;  // WU-UCT: simulations through this node whose value is still pending
    double value_sum = 0.0;
    bool evaluated = false;  // its rollout value is known (cached_value): a reused subtree keeps these across searches
    double cached_value = 0.0;
};

class Forest {
public:
    Forest(int num_trees, double c_puct)
        : c_puct_(c_puct),
          min_(num_trees, std::numeric_limits<double>::infinity()),
          max_(num_trees, -std::numeric_limits<double>::infinity()) {}

    // Adds tree's root (id = the current node count) and its children (the next K ids, in order); the Python side
    // appends the matching states in the same order.
    int add_root(int tree, int depth_left, const std::vector<double>& priors) {
        int id = static_cast<int>(nodes_.size());
        nodes_.push_back(Node{tree, -1, depth_left, 1.0});
        roots_.push_back(id);
        expand(id, priors);
        return id;
    }

    // The baseline's forced rounds 0..count-1, batched round-major: root child k of every tree.
    std::vector<int> forced(int count) const {
        std::vector<int> ids;
        for (int k = 0; k < count; ++k)
            for (int root : roots_) {
                const Node& r = nodes_[root];
                ids.push_back(r.first_child + k % r.num_children);
            }
        return ids;
    }

    // width PUCT selections per tree (tree-major), each adding an unobserved visit along its path (WU-UCT); with
    // width 1 the unobserved visit is released by the next backup before any selection reads it.
    std::vector<int> select(int width) {
        std::vector<int> ids;
        for (int root : roots_)
            for (int w = 0; w < width; ++w) {
                int node = root;
                while (nodes_[node].num_children > 0) node = best_child(node);
                ids.push_back(node);
                for (int n = node; n >= 0; n = nodes_[n].parent) nodes_[n].unobserved += 1;
            }
        return ids;
    }

    std::vector<int> depth_left(const std::vector<int>& ids) const {
        std::vector<int> out;
        for (int id : ids) out.push_back(nodes_[id].depth_left);
        return out;
    }

    // Which of ids already carry a rollout value (from this search or a reused subtree), and those values.
    std::pair<std::vector<bool>, std::vector<double>> cached(const std::vector<int>& ids) const {
        std::vector<bool> flags;
        std::vector<double> values;
        for (int id : ids) {
            flags.push_back(nodes_[id].evaluated);
            values.push_back(nodes_[id].cached_value);
        }
        return {flags, values};
    }

    // One round's results: priors holds K priors per *fresh* leaf with depth left (in ids order; a repeated leaf's
    // second copy is consumed and ignored), values one value per leaf, fresh[i] false for leaves whose value came
    // from the cache (already expanded, no priors). Expands, releases the unobserved visits, updates the min-max stats
    // and backs up, in ids order. Returns, per leaf, whether this call expanded it (its K children got the next
    // ids), so the Python side can append their states in the same order.
    std::vector<bool> backup(const std::vector<int>& ids, const std::vector<double>& values, const std::vector<double>& priors,
                             int num_children, bool release, const std::vector<bool>& fresh) {
        std::vector<bool> expanded(ids.size(), false);
        size_t cursor = 0;
        for (size_t i = 0; i < ids.size(); ++i) {
            int id = ids[i];
            if (nodes_[id].depth_left > 0 && fresh[i]) {
                if (nodes_[id].num_children == 0) {
                    expand(id, std::vector<double>(priors.begin() + cursor, priors.begin() + cursor + num_children));
                    expanded[i] = true;
                }
                cursor += num_children;
            }
            if (!nodes_[id].evaluated) {
                nodes_[id].evaluated = true;
                nodes_[id].cached_value = values[i];
            }
            if (release)
                for (int n = id; n >= 0; n = nodes_[n].parent) nodes_[n].unobserved -= 1;
            int tree = nodes_[id].tree;
            min_[tree] = std::min(min_[tree], values[i]);
            max_[tree] = std::max(max_[tree], values[i]);
            for (int n = id; n >= 0; n = nodes_[n].parent) {
                nodes_[n].value_sum += values[i];
                nodes_[n].visits += 1;
            }
        }
        return expanded;
    }

    std::vector<std::vector<int>> root_counts() const {
        std::vector<std::vector<int>> counts;
        for (int root : roots_) {
            const Node& r = nodes_[root];
            std::vector<int> c;
            for (int k = 0; k < r.num_children; ++k) c.push_back(nodes_[r.first_child + k].visits);
            counts.push_back(c);
        }
        return counts;
    }

    // The forest re-rooted at root child picks[tree] of every tree, for the next search: structure, priors and cached
    // rollout values are kept, every count is zero. Nodes are renumbered breadth-first (a node's children stay
    // contiguous); the second element maps each new id to its old id so the caller can carry the states over.
    std::pair<Forest, std::vector<int>> subtree(const std::vector<int>& picks) const {
        Forest out(static_cast<int>(roots_.size()), c_puct_);
        std::vector<int> old_ids;
        std::vector<std::pair<int, int>> queue;  // (old id, new id)
        for (size_t tree = 0; tree < roots_.size(); ++tree) {
            const Node& root = nodes_[roots_[tree]];
            int old = root.first_child + picks[tree] % root.num_children;
            out.roots_.push_back(static_cast<int>(out.nodes_.size()));
            out.nodes_.push_back(copy_of(nodes_[old], -1));
            old_ids.push_back(old);
            queue.emplace_back(old, out.roots_.back());
        }
        for (size_t q = 0; q < queue.size(); ++q) {
            int old = queue[q].first, id = queue[q].second;
            const Node& n = nodes_[old];
            if (n.num_children == 0) continue;
            int first = static_cast<int>(out.nodes_.size());
            for (int c = n.first_child; c < n.first_child + n.num_children; ++c) {
                queue.emplace_back(c, static_cast<int>(out.nodes_.size()));
                out.nodes_.push_back(copy_of(nodes_[c], id));
                old_ids.push_back(c);
            }
            out.nodes_[id].first_child = first;
            out.nodes_[id].num_children = n.num_children;
        }
        return {out, old_ids};
    }

    // The new search's root priors come from the model's prior at that cycle, as a fresh search's would.
    void set_root_priors(const std::vector<double>& flat) {
        size_t cursor = 0;
        for (int root : roots_) {
            const Node& r = nodes_[root];
            for (int k = 0; k < r.num_children; ++k) nodes_[r.first_child + k].prior = flat[cursor++];
        }
    }

    int size() const { return static_cast<int>(nodes_.size()); }
    int unobserved(int id) const { return nodes_[id].unobserved; }

private:
    static Node copy_of(const Node& n, int parent) {
        Node c{n.tree, parent, n.depth_left, n.prior};
        c.evaluated = n.evaluated;
        c.cached_value = n.cached_value;
        return c;
    }

    void expand(int id, const std::vector<double>& priors) {
        if (nodes_[id].num_children > 0 || nodes_[id].depth_left == 0) return;
        int first = static_cast<int>(nodes_.size());
        int tree = nodes_[id].tree, depth = nodes_[id].depth_left - 1;
        for (double p : priors) nodes_.push_back(Node{tree, id, depth, p});  // may reallocate: no references held
        nodes_[id].first_child = first;
        nodes_[id].num_children = static_cast<int>(priors.size());
    }

    double normalize(int tree, double value) const {
        return max_[tree] > min_[tree] ? (value - min_[tree]) / (max_[tree] - min_[tree]) : value;
    }

    int best_child(int id) const {
        const Node& n = nodes_[id];
        double sqrt_visits = std::sqrt(std::max(n.visits + n.unobserved, 1));
        int best = -1;
        double best_score = -std::numeric_limits<double>::infinity();
        for (int c = n.first_child; c < n.first_child + n.num_children; ++c) {
            const Node& child = nodes_[c];
            double q = child.visits ? normalize(n.tree, child.value_sum / child.visits) : 0.0;
            double score = q + c_puct_ * child.prior * sqrt_visits / (1 + child.visits + child.unobserved);
            if (score > best_score) {  // ties go to the first maximal child, as max() does in Python
                best_score = score;
                best = c;
            }
        }
        return best;
    }

    double c_puct_;
    std::vector<Node> nodes_;
    std::vector<int> roots_;
    std::vector<double> min_, max_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<Forest>(m, "Forest")
        .def(py::init<int, double>())
        .def("add_root", &Forest::add_root)
        .def("forced", &Forest::forced)
        .def("select", &Forest::select)
        .def("depth_left", &Forest::depth_left)
        .def("cached", &Forest::cached)
        .def("backup", &Forest::backup)
        .def("subtree", &Forest::subtree)
        .def("set_root_priors", &Forest::set_root_priors)
        .def("root_counts", &Forest::root_counts)
        .def("size", &Forest::size)
        .def("unobserved", &Forest::unobserved);
}
