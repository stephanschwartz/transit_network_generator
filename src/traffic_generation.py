
import itertools
import random

import networkx as nx


def make_trajectories_with_gravity(graph,
                                   num_centers,
                                   population_range,
                                   seed=None
                                   ):
    """
    We randomly give each center v a population p_v in the population_range.
    If w is a node with unit distance d to v, its population is increased
    by p_v / 2^d, for d <= 2 (parameters can be adjusted at the top).
    The demand on any o-d-pair (o, d) is then given by p_o * p_d.
    """
    if seed is not None:
        random.seed(seed)

    center_depth = 2  # nodes up to this distance also gain population boost from center node
    population_shrink_factor = 0.5  # node at distance d gets psb^d * center_population

    gravity_traffic = {e: 0 for e in graph.edges}

    population = {}
    forbidden_od = set()
    centers = network_k_means(graph, num_centers)

    for v in centers:
        popu = random.randint(*population_range)
        dists = nx.single_source_shortest_path_length(graph, v, cutoff=center_depth)
        for w, d in dists.items():
            population[w] = population.get(w, 0) + popu * population_shrink_factor ** d
        forbidden_center_od = itertools.combinations(dists.keys(), r=2)
        forbidden_od.update(forbidden_center_od)

    my_weight = 'weight' if nx.get_edge_attributes(graph, 'weight') else 'length'
    asp = dict(nx.all_pairs_dijkstra_path(graph, weight=my_weight))
    # aspl = dict(nx.all_pairs_dijkstra_path_length(graph, weight=my_weight))

    for o, d in itertools.combinations(population.keys(), r=2):
        if (o, d) in forbidden_od or (d, o) in forbidden_od:
            continue
        # dem = population[o] * population[d] / aspl[o][d]
        dem = population[o] * population[d]
        edge_path = zip(asp[o][d][:-1], asp[o][d][1:])
        for e in edge_path:
            my_e = e if e in gravity_traffic else e[::-1]
            gravity_traffic[my_e] += dem

    return gravity_traffic


def make_trajectories_random_od(graph,
                                od_targets_range,
                                od_demand_range,
                                leaf_od_rate,
                                leaf_od_scale,
                                seed=None
                                ):
    """
    We randomly choose origin-destination-pairs (starting from every
    node) with random demand and assume that drivers take shortest paths.
    If the traffic starting at leaves should be boosted, one can use the
    parameters leaf_od_rate (probability of leaf boost) and leaf_od_scale
    (boost factor).
    """
    if seed is not None:
        random.seed(seed)

    od_traffic = {e: 0 for e in graph.edges}

    my_weight = 'weight' if nx.get_edge_attributes(graph, 'weight') else 'length'
    two_level_network_paths = dict(nx.all_pairs_dijkstra_path(graph, weight=my_weight))
    leaves = set(v for v in graph.nodes if graph.degree[v] == 1)

    for o in graph.nodes():
        num_paths = random.randint(*od_targets_range)
        is_leaf = o in leaves
        ds = random.choices([v for v in graph.nodes() if v != o], k=num_paths)
        for d in ds:
            path = two_level_network_paths[o][d]
            demand = random.randint(*od_demand_range)
            if is_leaf and random.random() <= leaf_od_rate:
                demand *= leaf_od_scale
            if d in leaves and random.random() <= leaf_od_rate:
                demand *= leaf_od_scale

            prev_node = path[0]
            for current_node in path[1:]:
                e = (prev_node, current_node)
                my_e = e if e in od_traffic else e[::-1]
                od_traffic[my_e] += demand
                prev_node = current_node

    return od_traffic


def add_traffic(graph,
                # general
                frac_gravity_traffic=0.5,
                max_edge_traffic=100,
                seed=None,
                # gravity model
                num_centers=5,
                population_range=(50, 250),
                # od model
                od_targets_range=(1, 50),
                od_demand_range=(1, 100),
                leaf_od_rate=0.3,
                leaf_od_scale=25,
                ):
    if frac_gravity_traffic:
        gravity_traffic = make_trajectories_with_gravity(graph, num_centers, population_range, seed=seed)
        max_traf_gravity = max(gravity_traffic.values())
        gravity_scale = frac_gravity_traffic * 100 / max_traf_gravity
        gravity_traffic = {e: gravity_scale * gravity_traffic[e] for e in graph.edges}
    else:
        gravity_traffic = {e: 0 for e in graph.edges}

    if frac_gravity_traffic < 1:
        od_traffic = make_trajectories_random_od(graph, od_targets_range, od_demand_range, leaf_od_rate, leaf_od_scale,
                                                 seed=seed)
        max_traf_od = max(od_traffic.values())
        od_scale = (1 - frac_gravity_traffic) * 100 / max_traf_od
        od_traffic = {e: od_scale * od_traffic[e] for e in graph.edges}
    else:
        od_traffic = {e: 0 for e in graph.edges}

    sum_traf = {e: gravity_traffic[e] + od_traffic[e] for e in graph.edges}
    max_traf = max(sum_traf.values())
    scale_factor = max_edge_traffic / max_traf

    for e in graph.edges:
        graph.edges[e]['traffic'] = round(sum_traf[e] * scale_factor, 2)

    return


def network_k_means(graph, k, num_iter=20):

    centers = {v: [] for v in random.choices(list(graph.nodes), k=k)}
    aspl = dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight'))

    for iteration in range(num_iter):
        # assign nodes to closest center
        for v in graph.nodes:
            closest_center = min(centers, key=lambda c: aspl[c][v])
            centers[closest_center].append(v)

        # compute new center of each cluster
        new_centers = {}
        for c, v_list in centers.items():
            new_c = min(v_list, key=lambda v: sum(aspl[v][w] for w in v_list))
            new_centers[new_c] = []

        if all(c in new_centers for c in centers):
            break
        centers = new_centers

    return list(new_centers.keys())
