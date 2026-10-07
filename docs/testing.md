# Running the tests

After configuring the project with the Windows/CUDA/FLTK dependencies in
`CMakeLists.txt`, build and run the complete suite:

```powershell
cmake --build build-release --config Release -j 4
ctest --test-dir build-release -C Release --output-on-failure --timeout 60
```

For another build directory or configuration, substitute its path and use the
same configuration for both commands. GPU tests require a CUDA-capable device.
CMake supplies their evaluator resource directory from the source checkout, so
they do not depend on the test process's working directory.

## Coverage

- Betting legality and terminal chip utilities.
- Information-set keys distinguish the acting player's exact hand while ignoring
  the opponent's hand.
- River, turn, and flop public trees: parent/child links, public card transitions,
  chance probabilities, private hand domains, legal pair tables, action blocks,
  contiguous strategy tensors, and terminal payoff dimensions.
- Analytical check-only river EV: P0 wins all four private matchups, giving an EV
  of 1000 chips under the project's utility convention.
- Default record-computed CPU river EV; analytical record payoffs for wins,
  losses, ties, folds and pre-river all-ins. Record/precomputed payoff and CPU
  traversal parity across river, turn and flop, including decoded runouts.
- Multi-hand CPU learning and normalized strategies; convergence in both a known
  single-pair river game and a river game with hidden two-hand ranges.
- Best-response information privacy for both players, pair weights, zero-weight
  hands, public chance, and opponent-action-conditioned beliefs. Small synthetic
  games are checked against exhaustive pure-policy enumeration, and returned
  best-response policies are replayed to verify their reported values.
- GPU flattening, root learning signals, and CPU/GPU average strategy and EV
  agreement after one and 500 iterations. The agreement fixture forces two-pair
  chunks to exercise accumulation across chunks.

The street-tree tests share `tests/subgame_tree_checks.hpp` and build each tree
once. They use a small betting abstraction with pot-sized all-in bets and exact
public runouts. Timing benchmarks belong outside correctness tests.

## Current limits

The CPU terminal provider supports both `ValuePrecomputed` and `RecordComputed`
games. The GUI CPU path uses the default `RecordComputed` configuration, with
lazy board decoding and an all-in equity cache instead of a dense payoff table.
CPU/GPU agreement tests build `DebugComputed` games containing both terminal
representations, then run
the GPU in `RecordComputed` mode. The current chunked GPU traversal does not
implement `ValuePrecomputed` terminal loading. The default CPU record-mode
regression checks the solver path used by the GUI; it does not automate GUI clicks.

The best-response evaluator chooses one action per public decision and own-hand
bucket. It aggregates compatible opponent hands using pair probabilities and
opponent/chance reach, excluding the responder's own submitted strategy. This
supports multi-hand exploitability within the current exact-hand public-tree
representation. Values still depend on the game's terminal and chance semantics.

Turn/flop tests verify the public-tree representation and public chance
distribution. They do not establish correct private-card-conditioned runout
probabilities or solve accuracy. The turn/flop accuracy issue needs a separate
numerical regression fixture and solver fix.
