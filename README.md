# DDPG (Deep Deterministic Policy Gradient)

This repository contains a Python implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm and a few example scripts to run it on OpenAI Gym environments.

Files

- `DDPG.py` — core DDPG implementation (agent, actor/critic networks, replay buffer, training loop).
- `cart-pole-v0.py`, `pendulum-v0.py`, `acrobot-v1.py` — example scripts that run the agent on respective Gym environments.

Quick start

1. Create a virtual environment and install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run an example:

```powershell
python pendulum-v0.py
```

Notes

- The implementation assumes the use of OpenAI Gym and PyTorch (or NumPy-based networks depending on `DDPG.py`).
- Adjust hyperparameters inside `DDPG.py` or the example scripts for experimentation.

