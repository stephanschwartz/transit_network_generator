# Transit Network Generator

This code may help you to generate typical transit networks, e.g., for scientific purposes.
In many cases you have no or few real-world data or might not be allowed to share it.
Or you would like to generate instances of larger sizes to evaluate if your algorithms
still perform well.

The code provides three variants of graphs that are built, and two possibilities to
create random demand on the network. You can use, adapt, or expand it to your needs.


## Networks
There are three possibilities to build a random graph:

**Voronoi network:**
Create a Voronoi diagram and consider the graph that is spanned
by the ridges between the Voronoi cells. The is our basic Voronoi
graph with Euclidean edge lengths. Now, we can add additional
random leaves, split the edges to obtain a specified number of
nodes, or stretch each edge by a random factor.

**Two-level Voronoi network:**
The two level network consists of two nested Voronoi graphs.
The level 1 layer is a priority network where travelling is generally
faster.
This should model typical transit networks such as road networks 
(with primary and secondary roads) or public transit networks 
(with a train and a bus network).

**Tree:**
Create a tree from random points in a rectangle by using
Kruskal on complete graph with Euclidean edge weights.
Each node can have at most a given max degree.


## Demand/Traffic
There are two possibilities to create traffic:

**Gravity model:**
We randomly choose a given number of centers with a random population.
Also, each center increases the population of neighboring nodes.
The demand on any o-d-pair is then given by the product of the
populations of o and d.

**Random od-traffic:**
We randomly choose origin-destination-pairs (starting from every
node) with random demand and assume that drivers take shortest paths.
Parameters allow to boost the traffic starting at leave nodes.


## Exemplary networks

Two-level Voronoi:

![2lvlVoronoi](https://github.com/stephanschwartz/transit_network_generator/blob/main/doc/expl_two_level_voronoi.png
"Two-level Voronoi")

Tree:

![tree](https://github.com/stephanschwartz/transit_network_generator/blob/main/doc/expl_random_tree.png)


## Virtual Environment
The code was developed under Python 3.6 with the packages specified in requirements.txt.
