# UnderTheGun - Postflop Poker Solver 

## Demo (from Shark, but usage is nearly identical): https://youtu.be/3Rcd4PqS9L0

UnderTheGun is a completely free (and ad-free) open-source solver that implements state-of-the-art algorithms to solve Heads-Up (HU) poker. While other solvers exist, this project had two main goals:

1. Acceleration  – Implement GPU-Accelerated Counterfactual Regret Minimization (CFR), according to this research paper: https://arxiv.org/html/2408.14778v1
2. Accessibility – Allow anyone, even those unfamiliar with poker, to use the solver with ease.

Many features seen in other solvers have been intentionally omitted to reduce clutter and cognitive load. Bet/raise sizes vary by street:
- Flop: Bet 100%
- Turn/River: Bet 50%, 100% | Raise 250%, All in

These trade-offs were made to maintain a clean user experience.

> 🗂️ Installers (.zip files) are available on the Releases tab: https://github.com/quwin/UnderTheGun/releases

---

## 🎮 How to Use UnderTheGun

### Page 1: Initial Setup

Input the following:
- Stack Size
- Starting Pot
- Initial Min Bet
- All-In Threshold (default: 0.67)
- Pot Type (Single Raise, 3-bet, 4-bet)
- Your Position (SB, BB, UTG, UTG+1, MP, LJ, HJ, CO, BTN)
- Their Position
- Iterations (default: 100)
- Min Exploitability % (default: 0.1%, set to 0 to never stop early)
- Thread Count (default: CPU cores - 1)
- CFR Resolver (default: GPU) 

Options:
- Auto-import ranges – automatically loads ranges based on positions and pot type
- Force Donk Check – disables donk bets on flop (recommended for memory savings)

---

### Page 2: Board Selection
Click to select the board cards:
- 3 cards for flop (Or click "Random Flop" to auto-generate)
- 4 cards for turn
- 5 cards for river  
---

### Pages 3 & 4: Range Editing
Adjust your and villain's ranges.  
Auto-imported ranges are conservative (few 4-bet bluffs), so feel free to tune based on how aggressive/bluffy you or your opponent are.

---

### Page 5: Results
NOTE: results may take anywhere from 2s-4minutes based on range size, num iterations, and whether you are solving flop, turn, or river.
- View PIO-style strategy coloring
- Click a hand to see strategy for each combo
- Use dropdown to choose an action (check, bet, fold, etc.)

> ⚠️ If you select an action that occurs with low probability, solver outputs may be unreliable due to limited subtree exploration.

You’ll then be prompted to select the next board card if continuing the hand.  
Use Undo (bottom left) to go back one action or card.  
Use Back to return to inputs and solve a different game.

---

## 🧠 Flop Solving (RAM-Saving Mode)

Flop solving is the most memory-intensive part of the game tree. To make flop solves fit in a typical laptop's RAM, Shark applies a few flop-specific constraints that greatly reduce tree size and memory usage. The below ONLY applies for flop solves.

### How the solver works for flop games (and why)

1) Strategy compression to int16  
For flop solving, strategy arrays are stored in a compressed int16 format to reduce memory usage. (Regrets are already compressed for turn & flop.)

2) Max raises per street capped to 3 before force all-in  
To prevent raise-wars from exploding the game tree, each street is capped at 3 raises. After the cap is reached, further raising is replaced by a forced all-in option.

3) Force Donk Check is defaulted to true (highly recommended)  
By default, Shark disables donk betting on flop to reduce branching and help memory usage.  
Example: if flop goes p1 check → p2 raise/bet → p1 call, then on the turn p1 is forced to check (no donk lead).  
This can be toggled off, but is highly recommended to keep enabled for flop solves.

4) Flop action space is reduced to shrink tree size  
Flop only has bet options of 50% and 100%, and a single raise option of 100% to reduce tree size. Turn/river retains all 3 bet sizes and 2 raise sizes.

---

## 🪟 Windows Installation
1. Go to the Releases tab and download `utg_windows.zip`: https://github.com/24parida/shark-2.0/releases
2. Unzip the folder
3. Inside the folder, double-click underthegun.exe
4. Windows may warn you about an untrusted app — click More Options → Run Anyway

---

## 🔐 Security Note
The reason for having to trust the file is b/c to get a developer license is around $100/year for each platforms which I currently can't afford for just a side project :(.  
For anyone wrorried about security: this project is fully open source — feel free to inspect the code yourself.  
The build process is located in .github/workflows/new_ci.yml.

---

## 🛠 Developer Notes

Huge thanks to Anubhav Parida's Shark 2.0, which I used as a foundation with permission, including re-using their GUI:
https://github.com/24parida/shark-2.0/

### Key Improvements:
- Implemented GPU-Accelerated Counterfactual Regret Minimization
- 50% memory reduction via int16 strategy compression (enables larger flop solves)
- Collapse All-in calls to just Expected Value
- Per-street bet sizing: different bet/raise options for flop vs turn/river
- Bug fixes (e.g., proper chance node updates)
- Support for asymmetric ranges (Fossana required hero = villain)
- Added support for flop solving (not just turn)
- Improved reach probability propagation
- Fully functional GUI with oceanic blue theme
- Solve caching – skips redundant computations when re-exploring
- Undo history – navigate backwards through the game tree
- Strategy export – copy ranges in PIO-compatible format
- Thread count control – configure parallelism from the UI

### Base Algorithm:
- Counter Factual Regret Minimization
- Full GPU support w/ GPU-Accelerated Counterfactual Regret Minimization

Also uses HenryRLee’s PokerHandEvaluator for winner determination on showdowns:
https://github.com/HenryRLee/PokerHandEvaluator

---

## 👋 About Me

### About This Project
Hi, I'm Ethan Tran. I built this as a side project in order to improve my C++ skills, and also because I want to make poker more accessable!

### Future Optimizations I'd Like to Explore
- Improved optimizations
- Improved GUI design and overall UX
- Additional bet sizing configurations
- Preflop solver integration

Pull requests and forks are welcome!

If you found this project helpful or interesting, please star the repo or reach out 🙌

DM me with questions about the implementation or poker solving in general. I'd love to chat.
