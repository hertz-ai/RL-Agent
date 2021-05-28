import wikipediaapi
import networkx as nx
import networkx.classes.function as nxcf


class KG:
    def __init__(self):
        self.n_nodes = 40
        self.page_name = 'Chemistry'
        self.G = nx.DiGraph()

        self.makeGraph()
        self.pruneGraph()

        self.adjacencyMatrix = nx.linalg.graphmatrix.adjacency_matrix(self.G).A

    def makeGraph(self):
        def valid1(x):
            return ':' not in x and 'List' not in x and 'disambiguation' not in x

        def valid2(x):
            return valid1(x) and x in L[:self.n_nodes]

        wiki_api = wikipediaapi.Wikipedia(language='en')
        page = wiki_api.page(self.page_name)
        D = dict.fromkeys(filter(valid1, page.links.keys()))
        L = [self.page_name] + list(D.keys())
        for name in L[:self.n_nodes]:
            page = wiki_api.page(name)
            D[name] = list(filter(valid2, page.links.keys()))

        for name in L[:self.n_nodes]:
            if D[name]:
                for inner_name in D[name]:
                    if inner_name != name:
                        self.G.add_edge(inner_name, name)

    def pruneGraph(self):
        deg = {node: degree for node, degree in self.G.degree()}
        self.G.remove_nodes_from([node for node in self.G.nodes() if deg[node] < 2])

        to_remove = []
        edge_attrs = {}
        set_list = [tuple(sorted(edge)) for edge in self.G.edges()]
        duplicate_list = set(filter(lambda x: set_list.count(x) > 1, set_list))
        for edge in duplicate_list:
            u, v = edge
            top = max(edge, key=lambda x: deg[x])
            bottom = min(edge, key=lambda x: deg[x])
            if top == bottom:
                top, bottom = u, v
            edge_attrs[(top, bottom)] = {'weight': 2}
            to_remove.append((bottom, top))
        nxcf.set_edge_attributes(self.G, edge_attrs)
        self.G.remove_edges_from(to_remove)

    def showGraph(self):
        graph = nx.drawing.nx_pydot.to_pydot(self.G)
        graph.write_png('{}nodes{}Graph.png'.format(self.n_nodes, self.page_name))

    def checkCycles(self):
        cycles = list(nx.algorithms.cycles.simple_cycles(self.G))
        print('nos cycles', len(cycles))
        # adj = nx.linalg.graphmatrix.adjacency_matrix(self.G).A
        n_false_cycles = 0
        chem_in_cycle = 0
        for ind, cycle in enumerate(cycles):
            k = len(cycle)
            for i in range(k):
                if (cycle[i], cycle[(i + 1) % k]) in nxcf.get_edge_attributes(self.G, 'weight'):
                    n_false_cycles += 1
                    break
            else:
                if 'Chemistry' in cycle:
                    chem_in_cycle += 1
                else:
                    print(cycle)
        print('false cycles:', n_false_cycles)
        print('True cycles that have Chemistry in them:', chem_in_cycle)
        print(nxcf.number_of_edges(self.G))
        to_remove_2 = []
        for edge in nxcf.get_edge_attributes(self.G, 'weight'):
            if nxcf.get_edge_attributes(self.G, 'weight')[edge] == 2:
                to_remove_2.append(edge)
        self.G.remove_edges_from(to_remove_2)
        deg = {node: degree for node, degree in self.G.degree()}
        self.G.remove_nodes_from([node for node in self.G.nodes() if deg[node] == 0])
        print(nxcf.number_of_edges(self.G))


KG()
