# RL for matopeli.

## setup
```
conda create -n RL_snake python=3.7 

conda activate  RL_snake

pip install -r requirements.txt
```

## play the game 
### manual
```
python snake_game.py
```

### let the computer learn
```
python agent.py
```

### reload model
```
python agent.py -m <model_path>
```

Inspiration and codebase drawn from [python-engineer](https://github.com/python-engineer/snake-ai-pytorch) tutorials.