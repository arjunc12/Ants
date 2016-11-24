import networkx as nx
import pylab
import argparse
from kruskal import kruskal
from random import random, choice

ER_PROB = 0.03

def fig1_network():
    """ Manually builds the Figure 1 networks. """
    G = nx.grid_2d_graph(11,11)
    
    G.graph['name'] = 'fig1'
    G.graph['nests'] = [(3,2), (8, 3)]

    # horizontal edges.
    G.remove_edge((1,9),(2,9))
    G.remove_edge((3,9),(4,9))
    G.remove_edge((6,9),(7,9))
    G.remove_edge((8,9),(9,9))
    G.remove_edge((1,8),(2,8))
    G.remove_edge((8,8),(9,8))
    G.remove_edge((5,7),(6,7))
    G.remove_edge((9,7),(10,7))
    G.remove_edge((1,6),(2,6))
    G.remove_edge((5,6),(6,6))
    G.remove_edge((9,6),(10,6))
    G.remove_edge((1,5),(2,5))
    G.remove_edge((5,5),(6,5))
    G.remove_edge((7,5),(8,5))
    G.remove_edge((0,4),(1,4))
    G.remove_edge((2,4),(3,4))
    G.remove_edge((5,4),(6,4))
    G.remove_edge((7,4),(8,4))
    G.remove_edge((0,3),(1,3))
    #G.remove_edge((2,3),(3,3))
    G.remove_edge((3,3),(4,3)) # e
    G.remove_edge((3,2),(4,2))
    G.remove_edge((6,2),(7,2))
    G.remove_edge((0,1),(1,1))
    G.remove_edge((3,1),(4,1))
    G.remove_edge((8,1),(9,1))

    # vertical edges.
    G.remove_edge((1,0),(1,1))
    G.remove_edge((1,3),(1,4))
    G.remove_edge((1,5),(1,6))
    G.remove_edge((1,9),(1,8))
    #G.remove_edge((2,3),(2,4))
    G.remove_edge((2,5),(2,6))
    G.remove_edge((2,8),(2,9))
    G.remove_edge((3,1),(3,2))
    #G.remove_edge((3,3),(3,4))
    G.remove_edge((3,9),(3,10))
    G.remove_edge((4,1),(4,2))
    G.remove_edge((4,9),(4,10))
    G.remove_edge((5,4),(5,5))
    G.remove_edge((5,6),(5,7))
    G.remove_edge((6,2),(6,3))
    G.remove_edge((6,4),(6,5))
    G.remove_edge((6,6),(6,7))
    G.remove_edge((6,9),(6,10))
    G.remove_edge((7,2),(7,3))
    G.remove_edge((7,4),(7,5))
    G.remove_edge((7,9),(7,10))
    G.remove_edge((8,0),(8,1))
    G.remove_edge((8,4),(8,5))
    G.remove_edge((8,8),(8,9))
    G.remove_edge((9,0),(9,1))
    G.remove_edge((9,6),(9,7))
    G.remove_edge((9,8),(9,9))
    G.remove_edge((4,2),(4,3))
    G.remove_edge((4,3),(4,4))
    
    G.graph['init_path'] = []
    G.graph['init_path'].append(((7, 3), (8, 3)))
    G.graph['init_path'].append(((6, 3), (7, 3)))
    G.graph['init_path'].append(((5, 3), (6, 3)))
    G.graph['init_path'].append(((4, 3), (5, 3)))
    G.graph['init_path'].append(((3, 2), (3, 3)))

    #init_graph(G)
    
    return G

def simple_network():
    '''
    Manually builds a simple network with 3 disjoint paths between nest and target
    '''
    G = nx.grid_2d_graph(8, 6)
    
    G.graph['name'] = 'simple'
    G.graph['nests'] = [(0, 3), (7, 3)]
        
    for j in xrange(6):
        if j < 5:
            G.remove_edge((0, j), (0, j + 1))
            G.remove_edge((7, j), (7, j + 1))
        
        if j != 3:
            G.remove_edge((0, j), (1, j))
            G.remove_edge((6, j), (7, j))
    
    
    for j in [1, 2, 4]:
        for k in xrange(1, 6):
            G.remove_edge((k, j), (k + 1, j))
            if 2 <= k <= 5:
                try:
                    G.remove_edge((k, j), (k, j + 1))
                except:
                    pass
                try:
                    G.remove_edge((k, j), (k, j - 1))
                except:
                    pass
    
    G.graph['init_path'] = []
    for j in xrange(7):
        if j != 5:
            G.graph['init_path'].append(((j, 3), (j + 1, 3)))
    
    G.remove_edge((5, 3), (6, 3))
                                            
    #init_graph(G)
        
    return G
    
def simple_network_nocut():
    '''
    Manually builds a simple network with 3 disjoint paths between nest and target
    '''
    G = nx.grid_2d_graph(8, 6)
    
    G.graph['name'] = 'simple_nocut'
    G.graph['nests'] = [(0, 3), (7, 3)]
        
    for j in xrange(6):
        if j < 5:
            G.remove_edge((0, j), (0, j + 1))
            G.remove_edge((7, j), (7, j + 1))
        
        if j != 3:
            G.remove_edge((0, j), (1, j))
            G.remove_edge((6, j), (7, j))
    
    
    for j in [1, 2, 4]:
        for k in xrange(1, 6):
            G.remove_edge((k, j), (k + 1, j))
            if 2 <= k <= 5:
                try:
                    G.remove_edge((k, j), (k, j + 1))
                except:
                    pass
                try:
                    G.remove_edge((k, j), (k, j - 1))
                except:
                    pass
    
    G.graph['init_path'] = []
    for j in xrange(7):
        G.graph['init_path'].append(((j, 3), (j + 1, 3)))
                                            
    #init_graph(G)
        
    return G

def medium_network():
    '''
    Manually builds a simple network with 3 disjoint paths between nest and target
    '''
    G = nx.grid_2d_graph(8, 6)
    
    G.graph['name'] = 'medium'
    G.graph['nests'] = [(0, 3), (7, 3)]
        
    for j in xrange(6):
        if j < 5:
            G.remove_edge((0, j), (0, j + 1))
            G.remove_edge((7, j), (7, j + 1))
        
        if j != 3:
            G.remove_edge((0, j), (1, j))
            G.remove_edge((6, j), (7, j))
    
    
    for j in [1, 2, 4]:
        for k in xrange(1, 6):
            G.remove_edge((k, j), (k + 1, j))
            if 2 <= k <= 5:
                try:
                    G.remove_edge((k, j), (k, j + 1))
                except:
                    pass
                try:
                    G.remove_edge((k, j), (k, j - 1))
                except:
                    pass
    
    G.add_edge((3, 3), (3, 2))
    G.add_edge((3, 2), (4, 2))
    G.add_edge((4, 2), (5, 2))
    G.add_edge((5, 2), (6, 2))
    G.add_edge((6, 2), (7, 2))
    
    G.add_edge((3, 3), (3, 4))
    G.add_edge((3, 4), (3, 5))
    
    G.graph['init_path'] = []
    for j in xrange(7):
        if j != 5:
            G.graph['init_path'].append(((j, 3), (j + 1, 3)))
    
    G.remove_edge((5, 3), (6, 3))
                                            
    #init_graph(G)
        
    return G
    
def medium_network_nocut():
    '''
    Manually builds a simple network with 3 disjoint paths between nest and target
    '''
    G = nx.grid_2d_graph(8, 6)
    
    G.graph['name'] = 'medium_nocut'
    G.graph['nests'] = [(0, 3), (7, 3)]
        
    for j in xrange(6):
        if j < 5:
            G.remove_edge((0, j), (0, j + 1))
            G.remove_edge((7, j), (7, j + 1))
        
        if j != 3:
            G.remove_edge((0, j), (1, j))
            G.remove_edge((6, j), (7, j))
    
    
    for j in [1, 2, 4]:
        for k in xrange(1, 6):
            G.remove_edge((k, j), (k + 1, j))
            if 2 <= k <= 5:
                try:
                    G.remove_edge((k, j), (k, j + 1))
                except:
                    pass
                try:
                    G.remove_edge((k, j), (k, j - 1))
                except:
                    pass
    
    G.add_edge((3, 3), (3, 2))
    G.add_edge((3, 2), (4, 2))
    G.add_edge((4, 2), (5, 2))
    G.add_edge((5, 2), (6, 2))
    G.add_edge((6, 2), (7, 2))
    
    G.add_edge((3, 3), (3, 4))
    G.add_edge((3, 4), (3, 5))
    
    G.graph['init_path'] = []
    for j in xrange(7):
        G.graph['init_path'].append(((j, 3), (j + 1, 3)))
    
    #G.remove_edge((5, 3), (6, 3))
                                            
    #init_graph(G)
        
    return G

def square_grid_nocut(gridsize, gridname):
    G = nx.grid_2d_graph(gridsize, gridsize)
    G.graph['name'] = gridname + '_nocut'
    G.graph['nests'] = [(0, gridsize // 2), (gridsize - 1, gridsize // 2)]
    
    G.graph['init_path'] = []
    for i in xrange(gridsize - 1):
        G.graph['init_path'].append(((i, gridsize // 2), (i + 1, gridsize // 2)))
    #init_graph(G)
    
    return G

def square_grid(gridsize, gridname):
    G = nx.grid_2d_graph(gridsize, gridsize)
    G.graph['name'] = gridname
    G.graph['nests'] = [(0, gridsize // 2), (gridsize - 1, gridsize // 2)]
    
    G.remove_edge((gridsize // 2 - 1, gridsize // 2), (gridsize // 2, gridsize // 2))
    G.graph['init_path'] = []
    for i in xrange(gridsize - 1):
        if i != gridsize // 2 - 1:
            G.graph['init_path'].append(((i, gridsize // 2), (i + 1, gridsize // 2)))
    #init_graph(G)
    
    return G

def small_grid():
    return square_grid(7, 'small')
    
def small_grid_nocut():
    return square_grid_nocut(7, 'small_nocut')

def full_grid():
    '''
    Manually builds a full 11x11 grid graph, puts two nests at opposite ends of the middle
    of the grid, and removes the very middle edge
    '''
    G = nx.grid_2d_graph(11,11)
    
    G.graph['name'] = 'full'
    G.graph['nests'] = [(0, 5), (10, 5)]
    
    G.remove_edge((4, 5), (5, 5))
    
    G.graph['init_path'] = []
    for i in range(10):
        if i != 4:
            G.graph['init_path'].append(((i, 5), (i + 1, 5)))

    #init_graph(G)
    
    return G
    
def full_grid_nocut():
    G = nx.grid_2d_graph(11,11)
    
    G.graph['name'] = 'full_nocut'
    G.graph['nests'] = [(0, 5), (10, 5)]
    
    G.graph['init_path'] = []
    for i in range(10):
        G.graph['init_path'].append(((i, 5), (i + 1, 5)))

    #init_graph(G)
    
    return G

def er_network(p=0.5):
    G = nx.grid_2d_graph(11, 11)
    
    G.graph['name'] = 'er'
    nests = [(0, 5), (10, 5)]
    nest, target = nests
    G.graph['nests'] = nests
    
    node_list = G.nodes()
    
    for i in xrange(len(node_list)):
        for j in xrange(i):
            u, v = node_list[i], node_list[j]
            if u == nest and v == target:
                continue
            elif v == nest and u == target:
                continue
            elif u != v:
                if random() <= p:
                    G.add_edge(u, v)
                else:
                    if G.has_edge(u, v):
                        G.remove_edge(u, v)
    if not nx.has_path(G, nest, target):
        return None
    short_path = nx.shortest_path(G, nest, target)
    if len(short_path) <= 3:
        return None
    #print short_path
    idx = choice(range(1, len(short_path) - 1))
    #print idx
    G.graph['init_path'] = []
    G.remove_edge(short_path[idx], short_path[idx + 1])
    for i in xrange(idx):
        G.graph['init_path'].append((short_path[i], short_path[i + 1]))
    for i in xrange(idx + 1, len(short_path) - 1):
        G.graph['init_path'].append((short_path[i], short_path[i + 1]))
    #print P
        
    if not nx.has_path(G, nest, target):
        return None
            
    return G

def grid_span():
    grid = full_grid()
    spanning_tree1 = kruskal(grid.nodes(), sorted(grid.edges()))
    spanning_tree2 = kruskal(grid.nodes(), reversed(sorted(grid.edges())))
    grid.remove_edges_from(grid.edges())
    grid.add_edges_from(spanning_tree1 + spanning_tree2)
    grid.graph['name'] = 'grid_span'
    for i in range(10):
        if i != 4:
            grid.add_edge((i, 5), (i + 1, 5))
            grid.graph['init_path'].append(((i, 5), (i + 1, 5))) 
    return grid
    
def grid_span2():
    grid = grid_span()
    grid.graph['name'] = 'grid_span2'
    for i in range(10):
        if grid.has_edge((0, i), (0, i + 1)):
            grid.remove_edge((0, i), (0, i + 1))
    return grid
    
def grid_span3():
    grid = grid_span2()
    grid.graph['name'] = 'grid_span3'
    for i in range(10):
        if grid.has_edge((10, i), (10, i + 1)):
            grid.remove_edge((10, i), (10, i + 1))
    return grid

def get_graph(graph_name):
    G = None
    if graph_name == 'fig1':
        G = fig1_network()
    elif graph_name == 'simple':
        G = simple_network()
    elif graph_name == 'full':
        G = full_grid()
    elif graph_name == 'full_nocut':
        G = full_grid_nocut()
    elif graph_name == 'simple_nocut':
        G = simple_network_nocut()
    elif graph_name == 'small':
        G = small_grid()
    elif graph_name == 'tiny':
        G = tiny_grid()
    elif graph_name == 'medium':
        G = medium_network()
    elif graph_name == 'medium_nocut':
        G = medium_network_nocut()
    elif graph_name == 'grid_span':
        G = grid_span()
    elif graph_name == 'grid_span2':
        G = grid_span2()
    elif graph_name == 'grid_span3':
        G = grid_span3()
    elif graph_name == 'er':
        G = er_network(ER_PROB)
    return G
    
def main():
    graph_choices = ['fig1', 'full', 'simple', 'simple_weighted', 'simple_multi', \
                     'full_nocut', 'simple_nocut', 'small', 'tiny', 'medium', \
                     'medium_nocut', 'grid_span', 'grid_span2', 'grid_span3', 'er']
    parser = argparse.ArgumentParser()
    parser.add_argument("graphs", nargs='+', choices=graph_choices)
    
    args = parser.parse_args()
    graphs = args.graphs
    
    for graph in graphs:
        G = get_graph(graph)
        if G == None:
            continue
        path = G.graph['init_path']
        print path
        nests = G.graph['nests']
        pos = {}
        node_sizes = []
        node_colors = []
        for node in sorted(G.nodes()):
            pos[node] = (node[0], node[1])
            if node in nests:
                node_sizes.append(100)
                node_colors.append('m')
            else:
                node_sizes.append(10)
                node_colors.append('r')
        edge_widths = []
        edge_colors = []
        for u, v in sorted(G.edges()):
            if (u, v) in path or (v, u) in path:
                edge_widths.append(10)
                edge_colors.append('g')
            else:
                edge_widths.append(1)
                edge_colors.append('k')
        nx.draw(G, pos=pos, with_labels=False, nodelist=sorted(G.nodes()), \
                edgelist=sorted(G.edges()), width=edge_widths, edge_color=edge_colors, \
                node_size=node_sizes, node_color=node_colors)
        pylab.draw()
        #print "show"
        #PP.show()
        pylab.savefig("%s.png" % G.graph['name'], format='png')
        pylab.close()
    
    
if __name__ == '__main__':
    main()