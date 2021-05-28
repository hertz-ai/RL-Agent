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
            return valid1(x) and x in D[:self.n_nodes]

        wiki_api = wikipediaapi.Wikipedia(language='en')
        page = wiki_api.page(self.page_name)
        D = dict.fromkeys(filter(valid1, page.links.keys()))
        D = [self.page_name] + list(D.keys())
        for name in D[:self.n_nodes]:
            page = wiki_api.page(name)
            D[name] = list(filter(valid2, page.links.keys()))

        for name in D[:self.n_nodes]:
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
