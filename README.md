# GRACE
GRAC extension (Accepted at TKDD2022

# Instructions
Unzip all files in data

# Run 
python GRACE.py --kwargs

# Rum Examples
1) python GRACE.py --graph_type 'Hypergraph' --dataset cora --data_type cocitation --tol 0.1 --normalize l2 --alpha 0.5 --num_runs 2

2) python GRACE.py --graph_type 'Hypergraph' --dataset dblp --data_type coauthorship --tol 0.5 --normalize None --alpha 0.5 --num_runs 2

3) python GRACE.py --graph_type 'Hypergraph' --dataset cora --data_type coauthorship --tol 0.12 --normalize l2 --alpha 0.5 --num_runs 2
 
4)  python GRACE.py --graph_type 'Hypergraph' --dataset pubmed --data_type cocitation --tol 0.1 --normalize l2 --alpha 0.5 --num_runs 2

5) python GRACE.py --graph_type 'Hypergraph' --dataset citeseer --data_type cocitation --tol 0.1 --normalize None --alpha 0.5 --num_runs 2

6) python GRACE.py --graph_type 'Undirected' --dataset citeseer --data_type cocitation --tol 0.15 --normalize None --alpha 0.2 --num_runs 2

7) python GRACE.py --graph_type 'Undirected' --dataset wiki --data_type cocitation --tol 0.15 --normalize l1 --alpha 0.2 --num_runs 2

8) python GRACE.py --graph_type 'Undirected' --dataset cora --data_type cocitation --tol 0.15 --normalize l2 --alpha 0.2 --num_runs 2

9) python GRACE.py --graph_type 'Undirected' --dataset pubmed --data_type cocitation --tol 0.0 --normalize l1 --alpha 0.5 --num_runs 2

10) 
