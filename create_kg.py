import wikipediaapi
import networkx as nx
import networkx.classes.function as nxcf
import random
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt


class KG:
    def __init__(self):
        self.console = Console()

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
        self.G.remove_nodes_from(
            [node for node in self.G.nodes() if deg[node] < 2])

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

    def __repr__(self):
        table = Table(title='Nodes and their attributes')
        table.add_column('No')
        table.add_column('Name')
        for name in self.G.nodes(data=True)[list(self.G.nodes())[0]].keys():
            table.add_column(name)
        for ind, node in enumerate(self.G.nodes()):
            table.add_row(str(ind), str(node), *list(map(str, self.G.nodes()[node].values())))
        with self.console.capture() as capture:
            self.console.print(table)
        return capture.get()

    def showGraph(self):
        nx.draw_networkx(self.G,
                         pos=nx.spring_layout(self.G),
                         with_labels=True,
                         node_color='#E9CF24',
                         node_size=400)
        plt.show()

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
        self.G.remove_nodes_from(
            [node for node in self.G.nodes() if deg[node] == 0])
        print(nxcf.number_of_edges(self.G))

    def initializeScores(self, threshold: float = 0.35, score_low: float = 0.3, score_high: float = 0.8):
        def step(node):
            nonlocal threshold, score
            k = round(score + random.gauss(0, score / 10), 2)
            a = round(random.uniform(0, k), 2)
            self.G.nodes[node]['knowledge score'] = max(k, 0)
            self.G.nodes[node]['application score'] = max(a, 0)
            threshold += 0.05
            score -= random.uniform(0.01, min(0.3, score))
            if score > 0 and threshold < 1:
                for neighbour in list(nx.neighbors(self.G, node)):
                    if random.random() > threshold:
                        step(neighbour)

        attrs = {node: {'knowledge score': 0, 'application score': 0}
                 for node in self.G.nodes()}
        nx.set_node_attributes(self.G, attrs)

        score = random.uniform(score_low, score_high)
        start = [node for node in self.G.nodes() if self.G.in_degree(node) == 0]
        for node in start:
            if random.random() > threshold:
                step(node)


obj = KG()
obj.initializeScores()
obj.showGraph()
