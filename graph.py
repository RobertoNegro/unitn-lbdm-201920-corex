import os
import pickle
import pandas as pd
import networkx as nx
import randomcolor
from networkx.drawing.nx_pydot import write_dot

INPUT_LAYER_FILE = 'in_layers/layer%d_fit.pickle'
INPUT_MATRIX_FILE = 'in/top3kvariance_plus_literature_genesymbol_transposed_for_corex.txt'
OUTPUT_FILE_DOT = 'out_graph/graph_%dlayers.dot'
OUTPUT_FILE_SVG = 'out_graph/graph_%dlayers.svg'
OUTPUT_FILE_LINEAR_SVG = 'out_graph/graph_%dlayers_linear.svg'

G = nx.Graph()

layers = []
for i in range(0, 2):
    input_layer_file = INPUT_LAYER_FILE % i
    with open(input_layer_file, 'rb') as input_file:
        layers.append(pickle.load(input_file))

color_map = randomcolor.RandomColor().generate(count=(len(layers) + 1), luminosity='light')

input_matrix = pd.read_csv(INPUT_MATRIX_FILE, sep=' ', header=[0])
genes = input_matrix.columns
G.add_nodes_from(genes, style='filled', fillcolor=color_map[0])

for (index, layer) in enumerate(layers):
    n_latent_factors = len(layer.tcs)
    print('==== LAYER %d (%d latent factors) ====' % (index, n_latent_factors))
    for i in range(0, n_latent_factors):
        G.add_node('L%d_%d' % (index, i), style='filled', fillcolor=color_map[index + 1])
    clusters = layer.clusters()
    for (i, cluster) in enumerate(clusters):
        if index == 0:
            G.add_edge(genes[i], 'L%d_%d' % (index, cluster))
        else:
            G.add_edge('L%d_%d' % (index - 1, i), 'L%d_%d' % (index, cluster))

output_file_dot = OUTPUT_FILE_DOT % len(layers)
output_file_svg = OUTPUT_FILE_SVG % len(layers)
output_file_linear_svg = OUTPUT_FILE_LINEAR_SVG % len(layers)

write_dot(G, output_file_dot)
os.system('dot -Tsvg -o %s %s' % (output_file_linear_svg, output_file_dot))
os.system('dot -Ksfdp -Goverlap=false -Tsvg -o %s %s' % (output_file_svg, output_file_dot))
# -Goverlap=false #Gspline=true
