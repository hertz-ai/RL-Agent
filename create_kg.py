import wikipediaapi
import networkx as nx
import networkx.classes.function as nxcf
import random
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt


class KG:
    """
    in any wikipedia page, there will be a number of links.
    This code generates the knowledge graph by selecting the first n links on the primary page (called the inclusion list) and checking the connections between them.
    If page A links to page B (both of which must be in the inclusion list) then a directed edge is drawn from B to A denoting that to learn A, one must first learn B.
    Cycles are a problem since they make it impossible to define a tree-structured graph of the dependencies. Thus, we eliminate cyles.
    But, not all cyles denote cyclical dependencies: some might indicate topic-subtopic relationship.
    So, we designate the simplest - and possibly the most common - cycles which are the ones involving just 2 nodes as topic-subtopic edges, and remove them from the graph.

    This class also assigns knowledge and application scores to the nodes.
    We cannot do this arbitrarily since if concept A depends on concept B then one could not have learnt A before learning B.
    Thus, we start assigning values from nodes which do not have any dependencies and then move inwards, assigning lower and lower scores until either there are no more nodes on the path or the values of the scores have dwindled to 0.

    It is also possible to use this class to reset the values of the scores back to zero which might have been modified over the course of learning.
    """

    def __init__(self):
        self.console = Console()

        self.n_nodes = 40
        self.page_name = "Chemistry"
        self.G = nx.DiGraph()

    def makeGraph(self):
        """
        creates the basic graph by calling the wikipedia api
        """

        def valid1(x):
            """
            checks if it is fine to add x to the list of nodes

            Args:
                x (str): It is a link from a primary page

            Returns:
                bool: Check if the page represented by x is a valid descriptive page and not a list or disambiguation
            """
            return ":" not in x and "List" not in x and "disambiguation" not in x

        def valid2(x):
            """
            checks if it is fine to add an edge to or from x

            Args:
                x (str): It is a link from a page

            Returns:
                bool: Check if the page represented by x is a valid descriptive page and not a list or disambiguation. Also verifies that the page is in the inclusion list - only then an edge is made from it.
            """
            return valid1(x) and x in L[: self.n_nodes]

        wiki_api = wikipediaapi.Wikipedia(language="en")
        page = wiki_api.page(self.page_name)
        D = dict.fromkeys(filter(valid1, page.links.keys()))
        L = [self.page_name] + list(D.keys())
        for name in L[: self.n_nodes]:
            page = wiki_api.page(name)
            D[name] = list(filter(valid2, page.links.keys()))

        for name in L[: self.n_nodes]:
            if D[name]:
                for inner_name in D[name]:
                    if inner_name != name:
                        self.G.add_edge(inner_name, name)

    def pruneGraph(self):
        """
        removes edges that go from A to B and also from B to A.
        """
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
            edge_attrs[(top, bottom)] = {"weight": 2}
            to_remove.append((bottom, top))
        nxcf.set_edge_attributes(self.G, edge_attrs)
        self.G.remove_edges_from(to_remove)

    def makeAdjacencyMatrix(self):
        """
        creates the adjacency matrix of the graph. the sequence of nodes is given by graph.nodes().
        """
        self.adjMatrix = nx.linalg.graphmatrix.adjacency_matrix(self.G).A

    def __repr__(self):
        """
        this prints the string as a table of nodes and their names.

        Returns:
            str: it returns the str representation of the table
        """
        table = Table(title="Nodes and their attributes")
        table.add_column("No")
        table.add_column("Name")
        for name in self.G.nodes(data=True)[list(self.G.nodes())[0]].keys():
            table.add_column(name)
        for ind, node in enumerate(self.G.nodes()):
            table.add_row(
                str(ind), str(node), *list(map(str, self.G.nodes()[node].values()))
            )
        with self.console.capture() as capture:
            self.console.print(table)
        return capture.get()

    def showGraph(self):
        """
        it renders the graph as an image using matplotlib
        """
        nx.draw_networkx(
            self.G,
            pos=nx.spring_layout(self.G),
            with_labels=True,
            node_color="#E9CF24",
            node_size=400,
        )
        plt.show()

    def checkCycles(self):
        """
        checks for cycles in the graph. false cycles are those that are made of topic-subtopic relations.
        since chemistry is the main topic for all subtopics, it should be quite easy to eliminate cycles with chemistry in them. but for the 40 nodes that was not necessary.
        """
        cycles = list(nx.algorithms.cycles.simple_cycles(self.G))
        print("nos cycles", len(cycles))
        # adj = nx.linalg.graphmatrix.adjacency_matrix(self.G).A
        n_false_cycles = 0
        chem_in_cycle = 0
        for ind, cycle in enumerate(cycles):
            k = len(cycle)
            for i in range(k):
                if (cycle[i], cycle[(i + 1) % k]) in nxcf.get_edge_attributes(
                    self.G, "weight"
                ):
                    n_false_cycles += 1
                    break
            else:
                if "Chemistry" in cycle:
                    chem_in_cycle += 1
                else:
                    print(cycle)
        print("false cycles:", n_false_cycles)
        print("True cycles that have Chemistry in them:", chem_in_cycle)
        print(nxcf.number_of_edges(self.G))
        to_remove_2 = []
        for edge in nxcf.get_edge_attributes(self.G, "weight"):
            if nxcf.get_edge_attributes(self.G, "weight")[edge] == 2:
                to_remove_2.append(edge)
        self.G.remove_edges_from(to_remove_2)
        deg = {node: degree for node, degree in self.G.degree()}
        self.G.remove_nodes_from([node for node in self.G.nodes() if deg[node] == 0])
        print(nxcf.number_of_edges(self.G))

    def initializeScores(
        self, threshold: float = 0.35, score_low: float = 0.3, score_high: float = 0.8
    ):
        """
        set the scores starting from nodes that have no dependencies and moving towards more and more dependent nodes.

        Args:
            threshold (float, optional): . Defaults to 0.35.
            score_low (float, optional): the minimum possible score for non-dependent nodes. Defaults to 0.3.
            score_high (float, optional): the maximum possible score for non-dependent nodes. Defaults to 0.8.
        """
        def step(node):
            """
            recursively called funcion to step through the nodes and then call their numbers.

            Args:
                node (Graph nodes): nodes in the graph
            """
            nonlocal threshold, score
            k = round(score + random.gauss(0, score / 10), 2)
            a = round(random.uniform(0, k), 2)
            self.G.nodes[node]["knowledge score"] = max(k, 0)
            self.G.nodes[node]["application score"] = max(a, 0)
            threshold += 0.05
            score -= random.uniform(0.01, min(0.3, score))
            if score > 0 and threshold < 1:
                for neighbour in list(nx.neighbors(self.G, node)):
                    if random.random() > threshold:
                        step(neighbour)

        attrs = {
            node: {"knowledge score": 0, "application score": 0}
            for node in self.G.nodes()
        }
        nx.set_node_attributes(self.G, attrs)

        score = random.uniform(score_low, score_high)
        start = [node for node in self.G.nodes() if self.G.in_degree(node) == 0]
        for node in start:
            if random.random() > threshold:
                step(node)

    def resetScores(self):
        """
        setting the nodes' knowledge and application scores back to 0
        """
        for node in self.G.nodes():
            self.G.nodes[node]["knowledge score"] = 0
            self.G.nodes[node]["application score"] = 0
