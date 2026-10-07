#include "subgame_tree_checks.hpp"

int main() {
    return test_support::run([] { test_support::check_subgame_tree(4); });
}
