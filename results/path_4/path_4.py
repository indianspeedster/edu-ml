from graphviz import Digraph


G = Digraph(format='png')


nodes = ['Original_data', 'lower_bound', f'optimizer \n adamW', 'hyperparameter search \n scenario 2', "results"]
for node in nodes:
    G.node(node)


edges = [('Original_data', 'lower_bound'), ('lower_bound', f'optimizer \n adamW'), (f'optimizer \n adamW',  'hyperparameter search \n scenario 2'), ('hyperparameter search \n scenario 2', 'results')]
for edge in edges:
    G.edge(*edge)





G.render('path_4', cleanup=True, view=False)