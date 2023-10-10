#!/bin/bash
# following codes are assumed to be executed undet conda env "LERS*"

# different batch_size_by_HADMID
# python main.py --use_gpu --train --batch_size_by_HADMID=512
# python main.py --use_gpu --train --batch_size_by_HADMID=256
# python main.py --use_gpu --train --batch_size_by_HADMID=128

# different gnn_type
python main.py --use_gpu --train --gnn_type=GINEConv
python main.py --use_gpu --train --gnn_type=GENConv
python main.py --use_gpu --train --gnn_type=GATConv

# validating
# python main.py --use_gpu --val --val_model_state_dict="task=MIX_gnn_type=GENConv_batch_size_by_HADMID=128_loss=4.5933"
