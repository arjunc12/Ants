MIN_PHEROMONE = 0
DECAY_CHOICES = ['constant', 'linear', 'exp']

def edge_weight(G, u, v):
    '''
    computes the weight of an edge by summing up the weight of the units
    '''
    return sum(G[u][v]['units'])

def check_graph(G, min_pheromone=MIN_PHEROMONE):
    for u, v in G.edges_iter():
        weight = G[u][v]['weight']
        assert weight >= min_pheromone
        wt = 0
        for unit in G[u][v]['units']:
            assert unit > min_pheromone
            wt += unit
        assert wt == weight

def decay_units(G, u, v, decay, time=1, min_pheromone=MIN_PHEROMONE):
    '''
    decreases weight of each pheromone unit on edge (u,v) according to decay rate and time
    '''
    G[u][v]['units'] = subtract(G[u][v]['units'], decay * time)
    G[u][v]['units'] = G[u][v]['units'][where(G[u][v]['units'] > min_pheromone)]
    G[u][v]['units'] = list(G[u][v]['units'])

def decay_edges_linear(G, nonzero_edges, decay, time=1, min_pheromone=MIN_PHEROMONE):
    '''
    decays weight on all nonzero edges according to linear decay.  Every unit loses
    decay*time in weight until it reaches the minimum pheromone level.
    
    Returns all edges that are at the minimum pheromone level after decaying
    '''
    zero_edges = []
    edge_map = G.graph['edge_map']
    for i in nonzero_edges:
        u, v = edge_map[i]
        decay_units(G, u, v, decay, time)
        wt = edge_weight(G, u, v)
        assert wt >= min_pheromone
        G[u][v]['weight'] = wt
        if wt == min_pheromone:
            zero_edges.append(i)
    return zero_edges
    
def decay_graph_linear(G, decay, time=1, min_pheromone=MIN_PHEROMONE):
    '''
    decays all edges in the graph using linear decay
    '''
    assert decay > 0
    assert decay < 1
    for u, v in G.edges_iter():
        decay_units(G, u, v, decay, time)
        wt = edge_weight(G, u, v)
        assert wt >= min_pheromone
        G[u][v]['weight'] = wt

def decay_graph_exp(G, decay, time=1, min_pheromone=MIN_PHEROMONE):
    '''
    Decays the graph according to exponential decay.  Every edge reduces in weight by a
    specified proportion
    '''
    assert decay >= 0
    assert decay <= 1
    for u, v in G.edges_iter():
        before_weight = G[u][v]['weight']
        after_weight = before_weight * ((1 - decay) ** time)
        if before_weight == after_weight and decay != 0:
            G[u][v]['weight'] = min_pheromone
        else:
            G[u][v]['weight'] = after_weight
        G[u][v]['weight'] = max(G[u][v]['weight'], min_pheromone)
        assert G[u][v]['weight'] >= min_pheromone
        
def decay_edges_exp(G, nonzero_edges, decay, time=1, min_pheromone=MIN_PHEROMONE):
    '''
    decays graph according to exponential decay
    '''
    assert decay >= 0
    assert decay <= 1
    zero_edges = []
    edge_map = G.graph['edge_map']
    for i in nonzero_edges:
        u, v = edge_map[i]
        before_weight = G[u][v]['weight']
        after_weight = before_weight * ((1 - decay) ** time)
        if before_weight == after_weight  and decay != 0:
            G[u][v]['weight'] = min_pheromone
        else:
            G[u][v]['weight'] = after_weight
        G[u][v]['weight'] = max(G[u][v]['weight'], min_pheromone)
        wt = G[u][v]['weight']
        assert wt >= min_pheromone
        if wt == min_pheromone:
            zero_edges.append(i)
    return zero_edges
        
def decay_graph_const(G, decay, time=1, min_pheromone=MIN_PHEROMONE):
    '''
    decays graph according to constant decay.  Every edge loses a constant amount of weight
    '''
    assert decay > 0
    assert decay < 1
    for u, v in G.edges_iter():
        G[u][v]['weight'] -= decay * time
        G[u][v]['weight'] = max(G[u][v]['weight'], min_pheromone)
        assert G[u][v]['weight'] >= min_pheromone

def print_weights(G):
    '''
    prints out weights of all edges in the graph
    '''
    for u, v in G.edges_iter():
        print u, v, G[u][v]['weight'], G[u][v]['units']

def get_weights(G, start, candidates):
    '''
    Returns an array containing all the edge weights in the graph
    '''
    weights = map(lambda x : G[start][x]['weight'], candidates)
    return array(weights)
    
def get_decay_func(decay_type):
    decay_func = None
    if decay_type == 'linear':
        decay_func = decay_edges_linear
    elif decay_type == 'const':
        decay_func = decay_edges_const
    elif decay_type == 'exp':
        decay_func = decay_edges_exp
    else:
        raise ValueError('Invalid Decay Type')
    return decay_func