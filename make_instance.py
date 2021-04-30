import json

import networkx as nx
from networkx.readwrite import json_graph

import network_generation as ng
import traffic_generation as tg


class TransitNetwork:

    def __init__(self,
                 graph_params,
                 traffic_params,
                 seed=0,
                 from_load=False
                 ):

        self.graph_params = graph_params
        self.traffic_params = traffic_params
        self.seed = seed

        if not from_load:
            self.network = ng.make_random_graph(graph_params, seed)
            tg.add_traffic(self.network, **traffic_params, seed=seed)

    def print_stats(self):
        print('\nNetwork statistics:')
        n = self.network.number_of_nodes()
        e = self.network.number_of_edges()
        print(f'\tnumber of nodes: \t{n}\n\tnumber of edges: \t{e}')

    def save_instance_to_json(self, filepath):
        args = vars(self)
        params = {k: v for k, v in args.items() if not isinstance(v, nx.Graph)}

        json_instance = {
            'params': params,
            'network': json_graph.node_link_data(self.network)
        }

        with open(filepath, 'w') as f:
            json.dump(json_instance, f, indent=2)
        print(f'\nInstance written to {filepath}\n')


def load_transit_network_from_json(filepath):
    print(f'\nLoading Instance from file {filepath}...')
    with open(filepath, 'r') as f:
        json_instance = json.load(f)

    instance = TransitNetwork(**json_instance['params'], from_load=True)
    instance.network = json_graph.node_link_graph(json_instance['network'])
    return instance


def main():
    """
    There are three possibilities to build a random graph
    (exemplary input parameters are given below):

        * tree:
            Create a tree from random points in a rectangle by using
            Kruskal on complete graph with Euclidean edge weights.
            Each node can have at most a given max degree.
        * voronoi:
            Create a Voronoi diagram and consider the graph that is spanned
            by the ridges between the Voronoi cells. The is our basic Voronoi
            graph with Euclidean edge lengths. Now, we can add additional
            random leaves, split the edges to obtain a specified number of
            nodes, or stretch each edge by a random factor.
        * two_level_voronoi:
            The two level network consists of two nested Voronoi graphs.
            The level 1 layer is a priority network where travelling is generally
            faster (adjust with length_factor_level_1). Therefore, an additional
            edge attribute 'weight' is introduced.

    Then, there are two possibilities to create traffic:
        * gravity model:
            We randomly choose a given number of centers with a population
            in the given population_range. Also, each center increases the
            population of neighboring nodes (parameters for this are set in
            the according function).
            The demand on any o-d-pair (o, d) is then given by p_o * p_d
            (we do not divide by the distance between o and d).
        * random od-traffic:
            We randomly choose origin-destination-pairs (starting from every
            node) with random demand and assume that drivers take shortest paths.
            If the traffic starting at leaves should be boosted, one can use the
            parameters leaf_od_rate (probability of leaf boost) and leaf_od_scale
            (boost factor).
    """
    graph_params = {
        'variant': 'tree',
        'num_nodes': 200,
        'x_lim': 60,
        'y_lim': 100,
        'max_degree': 4,
        'edge_stretch_interval': (1, 1.5),
    }

    graph_params = {
        'variant': 'voronoi',
        'num_voronoi_points': 20,
        'x_lim': 60,
        'y_lim': 100,
        'random_leaves_frac': 0.7,
        'num_nodes': 200,
        'edge_stretch_interval': (1, 1.5),
    }

    graph_params = {
        'variant': 'two_level_voronoi',
        'n_level_1_voronoi_points': 30,
        'n_level_2_voronoi_points': 100,
        'x_lim': 100,
        'y_lim': 100,
        'random_leaves_frac_l1': 0,
        'random_leaves_frac_l2': 0.4,
        'n_level_1_nodes': 75,
        'num_nodes': 500,
        'edge_stretch_interval': (1, 1.5),
        'length_factor_level_1': 0.7,
    }

    traffic_params = {
        'num_centers': 250,
        'population_range': (50, 250),
        'od_targets_range': (1, 50),
        'od_demand_range': (1, 100),
        'leaf_od_rate': 0.8,
        'leaf_od_scale': 200,
        'max_edge_traffic': 100,
        'frac_gravity_traffic': 0.6,
    }

    seed = 3

    instance = TransitNetwork(graph_params=graph_params, traffic_params=traffic_params, seed=seed)

    ng.draw_graph(instance.network, traffic=True)


if __name__ == '__main__':

    main()
