# Independent postflop convergence reference

`test_postflop_convergence` checks exact-hand turn and flop games against a
separately constructed 16-by-16 normal-form payoff matrix. Neither CFR nor the
production best-response evaluator generates the reference matrix or equilibrium.

Both fixtures start with a 1000-chip pot and 1000 chips behind for each player.
P0 holds KhQh or 2d3c with relative frequencies 1:2. P1 holds QcQd or Th9h with
frequencies 3:1. The flop is As7hJh; the turn fixture adds Ts. Legal pair weights
are the range products, normalized over compatible pairs.

The betting game permits an opening check or shove. Facing a shove, the other
player folds or calls. After an opening check, P1 can check or shove and P0 can
fold or call. Checking through ends betting and runs out the board. No later
street bets, intermediate bet sizes, raises, rake, or suit compression are used.
Checkdown and called shoves use the record-computed exact equity path; terminal
utilities retain the project's subgame-net convention.

The independent matrix enumerates all 44 legal rivers or 990 unordered final
boards for each private pair using the host seven-card evaluator. It then
enumerates every pure policy: each player has four binary decisions, two public
decision states times two own-hand buckets. Payoffs for checks, folds, and called
shoves are calculated directly from these equities and chip amounts.

SciPy/HiGHS solves separate maximizing and minimizing linear programs offline.
The checked-in mixed-policy certificates in `tests/postflop_reference.hpp` are
verified at runtime against **every** opposing pure policy. This proves the
reference minimax value without requiring SciPy on test machines:

| Starting board | Certified P0 value, chips |
| --- | ---: |
| As7hJhTs | 482.9545454545455 |
| As7hJh | 257.8555252335740 |

The test converts each trained behavioral policy into a distribution over pure
policies and computes independent best-response bounds from the matrix. CPU and
GPU CFR/CFR+ must reach an EV within one chip of the reference and exploitability
below one chip, measured as half the upper/lower best-response gap. Production
EV and exploitability must agree with the independent calculation. Vanilla CFR
uses 50,000 iterations and a four-pair GPU chunk; CFR+ with linear averaging uses
20,000 iterations and partial three-pair GPU chunks. Each starting street is a
separate CTest case so longer convergence runs remain independently bounded.

## Regenerating the certificates

Build `test_postflop_convergence`, then export the independent matrix and use an
isolated Python environment for the developer-only SciPy dependency:

```powershell
& build-release/Release/test_postflop_convergence.exe --export-matrix build-release/postflop-matrices.txt
python -m venv build-release/reference-venv
& build-release/reference-venv/Scripts/python.exe -m pip install scipy
& build-release/reference-venv/Scripts/python.exe tests/generate_postflop_reference.py build-release/postflop-matrices.txt tests/postflop_reference.hpp
```

Rebuild and run the test after regenerating the header. The certificates are
tied to the exact board, ranges, weights, and betting options above; altering
the fixture requires regenerating and rechecking both minimax bounds.

These benchmarks establish convergence for two deliberately bounded hidden-hand
postflop games. They complement the explicit public-runout and cross-street chip
tests; they do not certify arbitrary multistreet betting trees. Wider ranges,
intermediate bet sizes, and raises need additional reference fixtures before
claiming accuracy across the entire solver.
