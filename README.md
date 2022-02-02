# RL for matopeli.

![img](/extra/shot.png?raw=true "example")


## setup
```
conda create -n RL_snake python=3.7 -y 
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

### reload model for learning
```
python agent.py -m <model_path>
```
### let the computer play
```
python agent.py -m <model_path> --play
```

Inspiration and codebase drawn from [python-engineer](https://github.com/python-engineer/snake-ai-pytorch) tutorials.