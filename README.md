# UnderTheGun - Postflop Poker Solver 

## Demo (from Shark, but usage is nearly identical): https://youtu.be/3Rcd4PqS9L0

<img width="1536" height="859" alt="image" src="https://github.com/user-attachments/assets/126511f5-1b57-4095-8c99-e208ec80770f" />

UnderTheGun is a completely free (and ad-free) open-source solver that implements state-of-the-art algorithms to solve Heads-Up (HU) poker. While other solvers exist, this project had two main goals:

1. Acceleration  – Implement GPU-Accelerated Counterfactual Regret Minimization (CFR), according to this research paper: https://arxiv.org/html/2408.14778v1
2. Accessibility – Allow anyone, even those unfamiliar with poker, to use the solver with ease.

Many features seen in other solvers have been intentionally omitted to reduce clutter and cognitive load. Bet/raise sizes are preset, and reraises are limited to one:
- Bet minimum, 50%, 100% | Raise 250% | All in

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
- Pot Type (Single Raise, 3-bet, 4-bet), used to automaticlly preset Ranges
- Your Position (SB, BB, UTG, UTG+1, MP, LJ, HJ, CO, BTN)
- Their Position
- Iterations (default: 100)
- Min Exploitability % (default: 0.1%, set to 0 to never stop early)
- ~~Thread Count (default: CPU cores - 1)~~  currently unimplemented
- CFR Resolver (default: GPU) 

Options:
- Auto-import ranges – automatically loads ranges based on positions and pot type 
- Force Donk Check – ~~disables donk bets on flop~~ (recommended for memory savings) currently unimplemented

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

## 🪟 Windows Installation
1. Go to the Releases tab and download `utg_windows.zip`: https://github.com/24parida/shark-2.0/releases
2. Unzip the folder
3. Inside the folder, double-click underthegun.exe
4. Windows may warn you about an untrusted app — click More Options → Run Anyway

---

## 🔐 Security Note
The reason for having to trust the file is b/c to get a developer license is around $100/year for each platforms which I currently can't afford for just a side project :(  
For anyone wrorried about security: this project is fully open source — feel free to inspect the code yourself.  

---

## 🛠 Developer Notes

Huge thanks to Anubhav Parida's Shark 2.0, which I used as a foundation with permission, including re-using their GUI:
https://github.com/24parida/shark-2.0/

### Key Improvements:
- Implemented GPU-Accelerated Counterfactual Regret Minimization
- Collapse All-in calls to just Expected Value
- Various Bug fixes 
- Support for asymmetric ranges
- Added support for flop solving (not just turn)
- Fully functional GUI with oceanic blue theme
- Solve caching – skips redundant computations when re-exploring
- Undo history – navigate backwards through the game tree
- Strategy export – copy ranges in PIO-compatible format

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
- Improved optimizations (especially auxillary storage optimizations)
- Improved GUI design and overall UX
- Additional bet sizing configurations
- Preflop solver integration

Pull requests and forks are welcome!

If you found this project helpful or interesting, please star the repo or reach out 🙌

DM me with questions about the implementation or poker solving in general. I'd love to chat.
