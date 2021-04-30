import itertools

import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
import numpy as np
from scipy.spatial import Voronoi


########################################################################################################################
#                                                                                                                      #
#                   Trees                                                                                              #
#                                                                                                                      #
########################################################################################################################

def make_random_tree(num_nodes,
                     x_lim,
                     y_lim,
                     max_degree=4,
                     edge_stretch_interval=None,
                     seed=None
                     ):
    """
    Create a tree from random points in a rectangle by using
    Kruskal on complete graph with Euclidean edge weights.
    Each node can have at most a degree of max_degree.
    """
    if seed is not None:
        np.random.seed(seed)

    points = make_random_points(num_nodes, x_lim, y_lim)
    dist = {(i, j): np.linalg.norm(points[i] - points[j]) for i, j in itertools.combinations(range(len(points)), r=2)}
    tree = max_degree_kruskal(dist, max_degree)
    pos = {i: list(p) for i, p in enumerate(points)}
    nx.set_node_attributes(tree, pos, name='pos')

    # stretch edge lengths
    ####################################
    if edge_stretch_interval is not None:
        random_stretch_edge_lengths(tree, stretch_interval=edge_stretch_interval, seed=seed)

    return tree


def max_degree_kruskal(dist, max_degree):
    """
    adjusted from networkx for max_degree
    """
    subtrees = nx.utils.UnionFind()
    edges = sorted((d, e[0], e[1]) for e, d in dist.items())
    tree = nx.Graph()
    max_degree_nodes = set()
    for wt, u, v in edges:
        if u in max_degree_nodes or v in max_degree_nodes:
            continue
        if subtrees[u] != subtrees[v]:
            tree.add_edge(u, v, length=round(wt, 2))
            for node in [u, v]:
                if tree.degree[node] == max_degree:
                    max_degree_nodes.add(node)
            subtrees.union(u, v)
    return tree


########################################################################################################################
#                                                                                                                      #
#                   Voronoi graphs                                                                                     #
#                                                                                                                      #
########################################################################################################################

def make_voronoi_graph(num_voronoi_points,
                       x_lim,
                       y_lim,
                       random_leaves_frac=0.4,
                       num_nodes=None,
                       edge_stretch_interval=None,
                       seed=None,
                       ):
    """
    Create a Voronoi diagram and consider the graph that is spanned by the ridges between
    the Voronoi cells. The is our basic Voronoi graph with Euclidean edge lengths. Now,
    we can add additional random leaves (according to random_leaves_frac), split the edges
    to obtain a specified number of nodes, or stretch each edge by a random factor.

    :param num_voronoi_points:
        int: number of points to build Voronoi regions
    :param x_lim:
        float: upper x-range bound for box in which we generate Voronoi points (lower bound=0)
    :param y_lim:
        float: upper y-range bound for box in which we generate Voronoi points (lower bound=0)
    :param random_leaves_frac:
        float: add this percentage of additional leaves to the voronoi graph
    :param num_nodes:
        int (or None): split edges such that the returned graph has num_nodes many nodes
    :param edge_stretch_interval:
        tuple of length 2 (or None): choose random stretch factor in this interval for each edge
    :param seed:
        random seed for reproducible results
    :return: networkx.Graph
    """
    if seed is not None:
        np.random.seed(seed)

    # basic Voronoi graph
    ####################################
    points = make_random_points(num_voronoi_points, x_lim, y_lim)
    vor = Voronoi(points)

    graph = nx.Graph()
    in_box_vertices = {i: v for i, v in enumerate(vor.vertices) if all(v >= (0, 0)) and all(v <= (x_lim, y_lim))}
    in_box_edges = [(u, v) for u, v in vor.ridge_vertices if u in in_box_vertices and v in in_box_vertices]
    my_edges = [(u, v, round(np.linalg.norm(in_box_vertices[u] - in_box_vertices[v]), 2)) for u, v in in_box_edges]
    graph.add_weighted_edges_from(my_edges, weight='length')
    nx.set_node_attributes(graph, in_box_vertices, name='pos')
    graph = nx.convert_node_labels_to_integers(graph)

    # additional leaves
    ####################################
    if random_leaves_frac:
        _num_nodes = graph.number_of_nodes()
        for i in range(int(_num_nodes * random_leaves_frac)):
            u = _num_nodes + i + 1
            u_pos = make_random_points(1, x_lim, y_lim)[0]
            v, d = find_nearest_node_without_intersection(graph, u_pos)
            graph.add_node(u, pos=u_pos)
            graph.add_edge(u, v, length=round(d, 2))

    # split edges for given num_nodes
    ####################################
    if num_nodes is not None:
        num_splits = num_nodes - graph.number_of_nodes()
        split_edges(graph=graph, num_splits=num_splits)

    # stretch edge lengths
    ####################################
    if edge_stretch_interval is not None:
        random_stretch_edge_lengths(graph, stretch_interval=edge_stretch_interval, seed=seed)

    return graph


########################################################################################################################
#                                                                                                                      #
#                   Two-level Voronoi networks                                                                         #
#                                                                                                                      #
########################################################################################################################

def make_two_level_voronoi_graph(n_level_1_voronoi_points,
                                 n_level_2_voronoi_points,
                                 x_lim,
                                 y_lim,
                                 random_leaves_frac_l1=0,
                                 random_leaves_frac_l2=0.6,
                                 n_level_1_nodes=None,
                                 num_nodes=None,
                                 edge_stretch_interval=None,
                                 length_factor_level_1=0.5,
                                 seed=None,
                                 ):
    """
    The two level network consists of two nested Voronoi graphs. The level 1 layer is a
    priority network where travelling is generally faster (adjust with length_factor_level_1).
    Therefore, an additional edge attribute 'weight' is introduced.

    :param n_level_1_voronoi_points:
        int: number of points to build level 1 Voronoi regions
    :param n_level_2_voronoi_points:
        int: number of points to build level 2 Voronoi regions
    :param x_lim:
        loat: upper x-range bound for box in which we generate Voronoi points (lower bound=0)
    :param y_lim:
        float: upper y-range bound for box in which we generate Voronoi points (lower bound=0)
    :param random_leaves_frac_l1:
        float: add this percentage of additional leaves to the level 1 voronoi graph
    :param random_leaves_frac_l2:
        float: add this percentage of additional leaves to the level 2 voronoi graph
    :param n_level_1_nodes:
        int (or None): split edges such that the level 1 graph has num_nodes many nodes
    :param num_nodes:
        int (or None): split edges such that the returned graph has num_nodes many nodes
    :param edge_stretch_interval:
        tuple of length 2 (or None): choose random stretch factor in this interval for each edge
    :param length_factor_level_1:
        float: travelling on level 1 graph is faster by this factor (a new edge attribute 'weight'
        is introduced)
    :param seed:
        random seed for reproducible results
    :return: networkx.Graph
    """
    
    level_1_layer = make_voronoi_graph(num_voronoi_points=n_level_1_voronoi_points, x_lim=x_lim, y_lim=y_lim,
                                       random_leaves_frac=random_leaves_frac_l1, num_nodes=None,
                                       edge_stretch_interval=None, seed=seed)
    level_2_layer = make_voronoi_graph(num_voronoi_points=n_level_2_voronoi_points, x_lim=x_lim, y_lim=y_lim,
                                       random_leaves_frac=random_leaves_frac_l2, num_nodes=None,
                                       edge_stretch_interval=None, seed=seed)

    # map level_1_layer to level_2_layer
    ####################################
    v2_list, pos2_list = zip(*level_2_layer.nodes(data='pos'))
    l1_to_l2_nodes = {}
    for v1, pos1 in level_1_layer.nodes(data='pos'):
        i = find_closest_point(pos1, pos2_list)
        l1_to_l2_nodes[v1] = v2_list[i]

    level_1_edges = set()
    for u, v in level_1_layer.edges:
        sh_path = nx.dijkstra_path(level_2_layer, l1_to_l2_nodes[u], l1_to_l2_nodes[v], weight='length')
        for e in zip(sh_path[:-1], sh_path[1:]):
            level_1_edges.add(tuple(sorted(e)))
    level_2_edges = set(e for e in level_2_layer.edges if e not in level_1_edges)

    two_level_network = level_2_layer
    level = {e: 1 + int(e in level_2_edges) for e in two_level_network.edges}
    nx.set_edge_attributes(two_level_network, level, name='level')

    # split level 1 edges
    ####################################
    if n_level_1_nodes is not None:
        num_splits = n_level_1_nodes - len(set(l1_to_l2_nodes.values()))
        split_edges(graph=two_level_network, num_splits=num_splits, no_split_edges=level_2_edges)

    # travelling on level_1_graph is faster
    ####################################
    weight = {(u, v): l * length_factor_level_1 if (u, v) in level_1_edges else l
              for u, v, l in two_level_network.edges(data='length')}
    nx.set_edge_attributes(two_level_network, weight, name='weight')

    # split level 2 edges
    ####################################
    if num_nodes is not None:
        num_splits = num_nodes - two_level_network.number_of_nodes()
        no_split_edges = [e for e in two_level_network.edges if two_level_network.edges[e]['level'] == 1]
        split_edges(graph=two_level_network, num_splits=num_splits, no_split_edges=no_split_edges)

    # stretch edge lengths
    ####################################
    if edge_stretch_interval is not None:
        random_stretch_edge_lengths(two_level_network, stretch_interval=edge_stretch_interval, seed=seed)

    return two_level_network


########################################################################################################################
#                                                                                                                      #
#                   general functions                                                                                  #
#                                                                                                                      #
########################################################################################################################

def make_random_graph(graph_params, seed=None):
    """
    Detect based on graph_params, which graph should be built and call the respective function.
    """
    make_graph_function = {
        'tree': make_random_tree,
        'voronoi': make_voronoi_graph,
        'two_level_voronoi': make_two_level_voronoi_graph
    }
    variant = graph_params['variant']
    assert variant in make_graph_function
    graph_params = {k: v for k, v in graph_params.items() if k != 'variant'}
    graph = make_graph_function[variant](**graph_params, seed=seed)
    pos = {v: [round(p[0], 4), round(p[1], 4)] for v, p in graph.nodes(data='pos')}
    nx.set_node_attributes(graph, pos, 'pos')

    return graph


def make_random_points(num_points, x_lim, y_lim):
    points = []
    for i in range(num_points):
        pos_x = round(np.random.random() * x_lim, 4)
        pos_y = round(np.random.random() * y_lim, 4)
        points.append((pos_x, pos_y))
    points = np.array(points)
    return points


def find_nearest_node_without_intersection(graph, u_pos):
    """
    Return the nearest node in graph to u_pos, such that the
    direct line between the two does not intersect any edge
    of the graph.
    """

    def ccw(A, B, C):
        """
        code from: https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
        (ccw stands for counterclockwise)
        """
        x, y = 0, 1
        return (C[y] - A[y]) * (B[x] - A[x]) > (B[y] - A[y]) * (C[x] - A[x])

    def intersect(A, B, C, D):
        """
        Returns true if line segments AB and CD intersect.
        code from: https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
        """
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    pos = dict(graph.nodes(data='pos'))
    dists = sorted([(np.linalg.norm(pos[v] - u_pos), v) for v in graph.nodes])
    while True:
        dist, v = dists.pop(0)
        for a, b in graph.edges:
            if a == v or b == v:
                continue
            if intersect(pos[a], pos[b], pos[v], u_pos):
                break
        return v, dist


def find_closest_point(point, other_points):
    """
    finds the index of the point in other_points closest to given point
    """
    dist = [(np.linalg.norm(point - p), i) for i, p in enumerate(other_points)]
    closest_point = min(dist)[1]
    return closest_point


def split_edges(graph, num_splits, no_split_edges=None):
    """
    This function splits the edges (except for no_split_edges) in graph num_splits times.
    Therefore, we store in lens_edge_nsplit a triple (l, e, n_split) for every
    possible split_edge. Here, e is the edge which we cut into n_split many equal
    length pieces, and l indicates the length of any of the pieces.
    Once we decide to split an edge, the new triple is
        (l * n_split / (n_split + 1), e, n_split + 1).
    """
    if no_split_edges is None:
        no_split_edges = []

    lens_edge_nsplit = sorted([(graph.edges[e]['length'], e, 1) for e in graph.edges() if e not in no_split_edges])

    for _ in range(num_splits):
        l, e, num = lens_edge_nsplit.pop()
        new_triple = (l * num / (num + 1), e, num + 1)
        lens_edge_nsplit.append(new_triple)
        lens_edge_nsplit.sort()

    next_node_id = max(graph.nodes) + 1
    for length, e, num in lens_edge_nsplit:
        if num > 1:
            # edge e is replaced by num edges
            e_dict = graph.edges[e]
            graph.remove_edge(*e)
            e_dict['length'] = round(length, 2)
            last_v = e[0]
            pos_u = np.array(graph.nodes[e[0]]['pos'])
            pos_v = np.array(graph.nodes[e[1]]['pos'])
            delta = (pos_v - pos_u) / num
            for i in range(num - 1):
                i_pos = list(pos_u + delta * (i + 1))
                graph.add_node(next_node_id, pos=i_pos)
                graph.add_edge(last_v, next_node_id, **e_dict)
                last_v = next_node_id
                next_node_id += 1
            graph.add_edge(last_v, e[1], **e_dict)


def random_stretch_edge_lengths(graph, edge_list=None, stretch_interval=(1, 1.5), seed=None):

    if seed is not None:
        np.random.seed(seed)
    if edge_list is None:
        edge_list = graph.edges()

    for e in edge_list:
        stretch_factor = np.random.uniform(*stretch_interval)
        graph.edges[e]['length'] = round(graph.edges[e]['length'] * stretch_factor, 2)
    return


def draw_graph(graph, traffic=False):
    pos = nx.get_node_attributes(graph, 'pos')
    nx.draw_networkx_nodes(graph, pos=pos, node_size=6, node_color='k')
    nx.draw_networkx_edges(graph, pos=pos, width=2)
    if traffic:
        edge_colors = [graph.edges[e]['traffic'] for e in graph.edges]
        edges = nx.draw_networkx_edges(graph, pos=pos, edge_color=edge_colors,
                                       edge_cmap=cm.get_cmap('rainbow'), width=4)
        plt.colorbar(edges)

    plt.axis('off')
    plt.show()

