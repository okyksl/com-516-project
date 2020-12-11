import os
import argparse
import numpy as np

folder = 'outputs/multirun/runner/solver=naive,dataset=g1,dataset.seed=1'
patterns = [
    'lmbd=0,',
    'lmbd=0.1,',
    'lmbd=0.2,',
    'lmbd=0.3,',
    'lmbd=0.4,',
    'lmbd=0.5,',
    'lmbd=0.6,',
    'lmbd=0.7,',
    'lmbd=0.8,',
    'lmbd=0.9,',
    'lmbd=1.0,']

n_states = [ [] for _ in patterns ] 
objectives = [ [] for _ in patterns ]

exps = [f.path for f in os.scandir(folder) if f.is_dir()]
for exp in exps:
    # detect group
    pattern = None
    for i in range(len(patterns)):
        if patterns[i] in exp:
            pattern = i
            break
    
    f = exp + '/runner.log'
    with open(f, 'r')  as file:
        lines = file.readlines()
        n_state = float(lines[-1][ lines[-1].find('|S|:')+5 : ])
        i = 2
        while lines[-i].find('Objective:') == -1:
            i +=1
        objective = float(lines[-i][ lines[-i].find('Objective:')+11 : ])
    n_states[pattern].append(n_state)
    objectives[pattern].append(objective)

for i, pattern in enumerate(patterns):
    print(pattern)
    print('-' * 40)
    ns = np.array(n_states[i])
    print('|S|:', np.mean(ns), '+-', np.std(ns))
    print('max |S|:', int(np.max(ns)), 'min |S|: ', int(np.min(ns)))
    print()

    objs = np.array(objectives[i])
    print('f(S):', np.mean(objs), '+-', np.std(objs))
    print('max f(S):', np.max(objs), 'min f(S):', np.min(objs))
    print()

    print('S:', ns)
    print('f(S):', objs)
    print('-' * 40)

    print()
