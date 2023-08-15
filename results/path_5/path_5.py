from graphviz import Digraph


G = Digraph(format='png')


nodes = ['k_fold_data', 'upper_bound', f'optimizer \n adamW', 'hyperparameter search \n scenario 2', "results"]
for node in nodes:
    G.node(node)


edges = [('k_fold_data', 'upper_bound'), ('upper_bound', f'optimizer \n adamW'), (f'optimizer \n adamW',  'hyperparameter search \n scenario 2'), ('hyperparameter search \n scenario 2', 'results')]
for edge in edges:
    G.edge(*edge)





G.render('path_5', cleanup=True, view=False)