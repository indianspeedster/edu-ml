from graphviz import Digraph


G = Digraph(format='png')


nodes = ['kfold_data', 'lower_bound', f'optimizer \n adamW', 'hyperparameter search \n scenario 2', "results"]
for node in nodes:
    G.node(node)


edges = [('kfold_data', 'lower_bound'), ('lower_bound', f'optimizer \n adamW'), (f'optimizer \n adamW', 'hyperparameter search \n scenario 2'), ('hyperparameter search \n scenario 2', 'results')]
for edge in edges:
    G.edge(*edge)





G.render('path_3', cleanup=True, view=False)