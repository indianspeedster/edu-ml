from graphviz import Digraph


G = Digraph(format='png')


nodes = ['Original_data', 'upper_bound', f'optimizer \n adamW', 'hyperparameter search \n scenario 1', "results"]
for node in nodes:
    G.node(node)


edges = [('Original_data', 'upper_bound'), ('upper_bound', f'optimizer \n adamW'), (f'optimizer \n adamW',  'hyperparameter search \n scenario 1'), ('hyperparameter search \n scenario 1', 'results')]
for edge in edges:
    G.edge(*edge)





G.render('path_6', cleanup=True, view=False)