import networkx as nx
from sys import argv

def print_fix_path(G, city1, city2):
    assert G.has_edge(city1, city2)
    G.remove_edge(city1, city2)
    if not nx.has_path(G, city1, city2):
        print city1, city2, "no fix path"
    else:
        print city1, city2, "fix path", nx.shortest_path(G, city1, city2)
    G.add_edge(city1, city2)

def subnetwork(G, city1, city2):
    sp = nx.shortest_path(G, city1, city2)
    print "short path", sp
    for k in xrange(len(sp) - 1):
        a, b = sp[k], sp[k + 1]
        #print a, b
        G.remove_edge(a, b)
        #print nx.has_path(G, a, b)
        G.add_edge(a, b)
    i = len(sp) / 2
    j = i + 1
    u, v = sp[i], sp[j]
    print "cut nodes", u, v
    G.remove_edge(u, v)
    sp2 = nx.shortest_path(G, city1, city2)
    print "short path 2", sp2
    fp = nx.shortest_path(G, u, v)
    print "fix path", fp

    fix_path_nodes = []
    for i in xrange(len(sp) - 1):
        x, y = sp[i], sp[i + 1]
        if not G.has_edge(x, y):
            continue
        print_fix_path(G, x, y)

    subnet_nodes = sp + sp2 + fp + fix_path_nodes
    subnet_nodes = list(set(subnet_nodes))
    print subnet_nodes
    G.add_edge(u, v)
    subnet = G.subgraph(subnet_nodes)
    print subnet.edges()

def load_eroad():
    road_file = 'road_datasets/road_europe.net'
    city_dict = {}
    G = nx.Graph()
    for line in open(road_file):
        if line[0] == '*':
            continue
        line = line.strip('\n')
        line_items = None
        if '\"' in line:
            line_items = line.split('\"')
            assert len(line_items) == 3
            label = int(line_items[0])
            city = line_items[1]
            city_dict[label] = city
        else:
            line_items = line.split()
            assert len(line_items) == 2
            u, v = map(int, line_items)
            city1, city2 = city_dict[u], city_dict[v]
            G.add_edge(city1, city2)

    return G

def main():
    G = load_eroad()
    city1, city2 = argv[1:]
    subnetwork(G, city1, city2)

if __name__ == '__main__':
    main()
