#!/bin/bash
# following codes are assumed to be executed undet conda env "LERS*"

# different batch_size_by_HADMID
# python main.py --batch_size_by_HADMID=512
# python main.py --batch_size_by_HADMID=256
# python main.py --batch_size_by_HADMID=128

# different gnn_type
# python main.py --gnn_type=GINEConv --batch_size_by_HADMID=128
python main.py --gnn_type=GENConv --batch_size_by_HADMID=128
# python main.py --gnn_type=GATConv --batch_size_by_HADMID=128

# python main.py --gnn_type=GENConv --batch_size_by_HADMID=128 --top50_items