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
- Private-card-conditioned turn/flop chance: a guaranteed royal flush retains
  all 1000 chips, and mixed-hand check-only EVs match independent enumeration
  of all 44 legal rivers or 990 legal final boards per private pair.
- Weighted public chance with different blockers per pair, including analytic
  best-response values, counterfactual regret and average-strategy weights.
- Turn/flop CPU/GPU learning signals and average strategies across one-pair
  and partial three-pair chunks. GPU regret sums use unnormalized uniform pair
  weights, so comparisons divide them by the total pair count.
- Collapsed flop/turn all-in equities, including turn all-ins reached from a
  flop, and rejection of suit compression without private-hand remapping.

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

Public chance edges store the board-only distribution. CPU, GPU and profile/
best-response evaluation exclude the four private cards and renormalize legal
outcomes per pair. Exact turn transitions consequently use 1/44 rather than
1/48; flop transitions use 1/45 followed by 1/44. The GUI and benchmark use exact
runouts. The builder rejects suit-isomorphic abstractions until their private
hands can be remapped correctly; exact trees can use more memory.

These regressions establish runout/equity semantics and CPU/GPU agreement for
small fixtures. Independently certified turn/flop convergence benchmarks are
described in [postflop-reference.md](postflop-reference.md). They require CPU/GPU
CFR and CFR+ to reach sub-one-chip exploitability in two restricted betting
games, with reference probabilities and best-response bounds checked independently
of production traversal. They do not establish convergence for arbitrary large
flop games or validate every betting abstraction. The GPU's `last_root_value_p0`
stats field remains unpopulated; tests check its learning signals and evaluate
its returned policies through the independently checked profile evaluator.

Cross-street chip tests preserve completed-street contributions while resetting
street-local call/raise commitments, and check fold/loss/win/tie utilities plus
record-computed CPU/GPU learning after a non-all-in turn bet and river fold.

Range-product weights are retained for each legal hand pair and used by default
in CPU training, GPU counterfactual/average reach, and profile/best-response
evaluation. Empty weights on manually built games continue to mean uniform.

CPU CFR+ aggregates a complete iteration of counterfactual regret before
clipping at zero. Cancellation, private-pair permutations, nonuniform weights,
and CPU/GPU CFR+ with linear averaging are covered by dedicated regressions.

CPU/GPU average policies normalize the actual accumulated action mass, avoiding
drift between action sums and the separate weight counter. Repeated accumulation
of thirds over long runs must still produce valid probabilities summing to one.
