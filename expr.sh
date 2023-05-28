#!/bin/bash
# following codes are assumed to be executed undet conda env "LERS*"

# different gnn_type
python main.py --gnn_type=GINEConv
python main.py --gnn_type=GENConv
python main.py --gnn_type=GATConv

# different batch_size_by_HADMID
python main.py --batch_size_by_HADMID=512 --gnn_type=GINEConv
python main.py --batch_size_by_HADMID=256
python main.py --batch_size_by_HADMID=128

# different max_timestep
python main.py --max_timestep=10 --batch_size_by_HADMID=512 --gnn_type=GINEConv
python main.py --max_timestep=20
python main.py --max_timestep=30
python main.py --max_timestep=50