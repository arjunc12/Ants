import networkx as nx
import pylab

def makes_cycle(u, v, node_to_forest):
    #checks if adding an edge would make a cycle in a graph
    #This occurs if G has no cycles and two vertices are in a connectec component
    f1 = node_to_forest[u]
    f2 = node_to_forest[v]
    return f1 == f2
 
def combine_forests(u, v, node_to_forest, forest_to_nodes, forest_to_edges):
    '''
    combines two spanning forests into a single spanning forests.
    picks one forest and assigns to it all nodes and edges in other forests 
    '''
    f1 = node_to_forest[u]
    f2 = node_to_forest[v]
    assert f1 != f2
    new_forest = min(f1, f2)
    old_forest = max(f1, f2)
    update_nodes = forest_to_nodes[old_forest]
    update_edges = forest_to_edges[old_forest]
    for node in update_nodes:
        node_to_forest[node] = new_forest
    forest_to_nodes[new_forest] += update_nodes
    forest_to_edges[new_forest] += update_edges
    forest_to_edges[new_forest].append((u, v))
    del forest_to_nodes[old_forest]
    del forest_to_edges[old_forest]
    
def kruskal(nodes, edges):
    '''
    runs kruskal's minimum spanning forest algorithm

    iterate through edges; if an egdge can be added without creating
    a cycle, add it. at the end there will be a minimum spanning
    forrest
    '''
    node_to_forest = {}
    forest_to_nodes = {}
    forest_to_edges = {}
    
    for i, u in enumerate(nodes):
        node_to_forest[u] = i
        forest_to_nodes[i] = [u]
        forest_to_edges[i] = []
        
    for u, v in edges:
        if not makes_cycle(u, v, node_to_forest):
            combine_forests(u, v, node_to_forest, forest_to_nodes, forest_to_edges)
            
    return sum(forest_to_edges.values(), [])
    
if __name__ == '__main__':
    G = nx.grid_2d_graph(11, 11)
    mst = kruskal(G.nodes(), sorted(G.edges()))
    G.remove_edges_from(G.edges())
    G.add_edges_from(mst)
    pos = {}
    for node in G.nodes():
        pos[node] = (node[0], node[1])
    nx.draw(G, pos=pos, with_labels=False)
    pylab.draw()
    pylab.savefig("%s.png" % 'kruskal', format='png')
    pylab.close()
            
