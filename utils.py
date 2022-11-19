import pandas as pd
import numpy as np
from itertools import combinations

def polish_ranking_data(df):
  
    df['Score'] = df['Score'].str.split('.').apply(lambda x: x[0]).astype(int)

    return df

def logistic(x):
    return 1/(1 + np.exp(-x))

def goals_to_probs(goals):

    team1_wins = np.sum(goals[:,0] > goals[:,1])
    team2_wins = np.sum(goals[:,0] < goals[:,1])
    draw = np.sum(goals[:,0]==goals[:,1])

    results = np.array([team1_wins, draw, team2_wins])

    return results / results.sum()

def simulate_match(teams, data, **config):

    score_name = config.get('score_name', 'Score')
    norm = config.get('normalization_factor', 100),
    n_trials = config.get('n_trials', 500)
    verbose = config.get('verbose', False)

    dd = data[ data['Team'].isin(teams) ].sort_values(by=score_name)[['Team',score_name]]
    teams = dd['Team'].values
    scores = dd[score_name].values

    if verbose:
        print('###########################')
        print('Teams:', teams)
        print('Scores:', scores)

    delta = np.diff(scores)[0]
    if verbose:
        print('delta:', delta)

    p = logistic(delta/norm)[0]
    if verbose:
        print('logistic(delta):', p)

    ps = np.array([1-p, p])
    if verbose:
        print('logit scores:', ps)

    to_lambda = config.get('to_lambdas', lambda x: x)
    lambdas = to_lambda(ps)
    if verbose:
        print('lambdas:', lambdas)

    goals = np.random.poisson(lam=lambdas, size=(n_trials,2))
    probs = goals_to_probs(goals)
    if verbose:
        print('probabilities:', probs)
    
    output = {
        'teams': teams,
        'normalized_score_diff': delta/norm[0],
        'goals': goals,
        'probabilities': probs
    }
    
    return output

def simulate_group(data, group_name, **config):

    print(f'********* GROUP {group_name} ***********')
    
    outputs = []

    hack_for_draw = config.get('hack_for_draw',None)
    
    matches = list(combinations(data[ data['Group']==group_name ]['Team'].tolist(),2))

    team_to_points = { i:0 for i in data[ data['Group']==group_name ]['Team'].tolist() }

    for match in matches:
        
        match_output = simulate_match(match, data, **config)
        match_output['group'] = group_name
        outputs.append(match_output)
        
        results = [match_output['teams'][0],'Draw',match_output['teams'][1]]

        outcome = results[ match_output['probabilities'].argmax() ]
        if hack_for_draw is not None:
            if hack_for_draw(match_output['probabilities']):
                outcome = 'Draw'

        print('{0} - {1} --> {2} ({3:.1f}% / {4:.1f}% / {5:.1f}%)'.format(
            match_output['teams'][0],
            match_output['teams'][1],
            outcome,
            100*match_output['probabilities'][0],
            100*match_output['probabilities'][1],
            100*match_output['probabilities'][2]
            ))

        team_to_points[match_output['teams'][0]] += np.sum(np.array([3, 1, 0]) * match_output['probabilities'])
        team_to_points[match_output['teams'][1]] += np.sum(np.array([0, 1, 3]) * match_output['probabilities'])

    print('-- Final standing:')
    sorted_group = {k: v for k, v in sorted(team_to_points.items(), key=lambda item: item[1], reverse=True)}
    sorted_group['group'] = group_name
    print(sorted_group)
    
    return outputs, sorted_group

def logit_score_to_lambda(x, params):
    
    output = np.zeros(len(x))
    deg = len(params)-1
    for i,pi in enumerate(params):
        output += pi * x**(deg-i)
    
    return output

def extract(o):
    
    df = pd.DataFrame(o)
    
    df['p_best_wins'] = df['probabilities'].apply(lambda x: x[2])
    df['p_worst_wins'] = df['probabilities'].apply(lambda x: x[0])    
    df['p_draw'] = df['probabilities'].apply(lambda x: x[1])
    
    return df