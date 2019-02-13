import numpy as np

def read_graph(graph_filename):
    # p -> a : 0
    # a -> p : 1
    # p -> c : 2
    # c -> p : 3
    # p -> t : 4
    # t -> p : 5
    #graph_filename = '../data/dblp/dblp_triple.dat'

    relations = set()
    nodes = set()
    graph = {}

    with open(graph_filename) as infile:
        for line in infile.readlines():
            source_node, target_node, relation = line.strip().split(' ')
            source_node = int(source_node)
            target_node = int(target_node)
            relation = int(relation)

            nodes.add(source_node)
            nodes.add(target_node)
            relations.add(relation)

            if source_node not in graph:
                graph[source_node] = {}

            if relation not in graph[source_node]:
                graph[source_node][relation] = []

            graph[source_node][relation].append(target_node)

    n_node = len(nodes)
    #print relations
    return n_node, len(relations), graph

def str_list_to_float(str_list):
    return [float(item) for item in str_list]

def read_embeddings(filename, n_node, n_embed):

    embedding_matrix = np.random.rand(n_node, n_embed)
    i = -1
    with open(filename) as infile:
        for line in infile.readlines()[1:]:
            i += 1
            emd = line.strip().split()
            embedding_matrix[int(emd[0]), :] = str_list_to_float(emd[1:])
    return embedding_matrix

if __name__ == '__main__':
    n_node, n_relation, graph  = read_dblp_graph()

    #embedding_matrix = read_embeddings('../data/dblp/rel_embeddings.txt', 6, 64)
    print graph[1][1]
