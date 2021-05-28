import networkx as nx
import random
from rich.console import Console
from rich.table import Table
import importlib.util
from create_kg import KG


# spec = importlib.util.spec_from_file_location(
#     "KG", r"D:\Mimisbrunnr\Github Repositories\RL-Agent\create_kg.py")
# foo = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(foo)
# KG = foo.KG

console = Console()


def step(node):
    global threshold, score
    k = round(score + random.gauss(0, score / 10), 2)
    a = round(random.uniform(0, k), 2)
    G.nodes[node]['knowledge score'] = k
    G.nodes[node]['application score'] = a
    threshold += 0.05
    score -= random.uniform(0.01, min(0.3, score))
    if score > 0 and threshold < 1:
        for neighbour in list(nx.neighbors(G, node)):
            if random.random() > threshold:
                step(neighbour)


for i in range(1):
    G = KG().G
    attrs = {node: {'knowledge score': 0, 'application score': 0} for node in G.nodes()}
    nx.set_node_attributes(G, attrs)

    table = Table(title='K and A scores')
    table.add_column('node')
    table.add_column('knowledge score')
    table.add_column('application score')

    threshold = 0.35
    score = random.uniform(0.5, 0.8)

    # adj = nx.adjacency_matrix(G).A
    start = [node for node in G.nodes() if G.in_degree(node) == 0]
    for node in start:
        if random.random() > threshold:
            step(node)

    for node in G.nodes():
        L = [str(x) for x in G.nodes[node].values()]
        table.add_row(str(node), L[0], L[1])
    console.print(table)

    # l1 = {node: str(ind + 1) + '\n' + str(G.nodes[node]['knowledge score']) + ',' + str(
    #     G.nodes[node]['application score']) for ind, node in enumerate(G.nodes())}
    # nx.draw(G, labels=l1, node_size=4000, node_color='#dddddd', pos=nx.spectral_layout(G))
    # plt.savefig(r'D:\Mimisbrunnr\Github Repositories\RL-Agent\Trials\op{}.png'.format(i))
    # plt.clf()
