#!/bin/bash
# following codes are assumed to be executed undet conda env "LERS*"

### different batch_size_by_HADMID ###
# python main.py --use_gpu --train --batch_size_by_HADMID=512
# python main.py --use_gpu --train --batch_size_by_HADMID=256
# python main.py --use_gpu --train --batch_size_by_HADMID=128

### different gnn_type ###
# python main.py --use_gpu --train --gnn_type=GINEConv
# python main.py --use_gpu --train --gnn_type=GENConv
# python main.py --use_gpu --train --gnn_type=GATConv

# python main.py --use_gpu --train --gnn_type=GINEConv
# python main.py --use_gpu --train --gnn_type=GENConv
# python main.py --use_gpu --train --gnn_type=GATConv

# python main.py --use_gpu --train --gnn_type=GINEConv
# python main.py --use_gpu --train --gnn_type=GENConv
# python main.py --use_gpu --train --gnn_type=GATConv

# python main.py --use_gpu --train --gnn_type=GINEConv
# python main.py --use_gpu --train --gnn_type=GENConv
# python main.py --use_gpu --train --gnn_type=GATConv

# python main.py --use_gpu --train --gnn_type=GINEConv
# python main.py --use_gpu --train --gnn_type=GENConv
# python main.py --use_gpu --train --gnn_type=GATConv

### validating ###
# python main.py --use_gpu --val --val_num=1 --gnn_type=GINEConv --val_model_state_dict="task=MIX_gnn_type=GINEConv_batch_size_by_HADMID=128_loss=7.8410"
# python main.py --use_gpu --val --val_num=1 --gnn_type=GATConv  --val_model_state_dict="task=MIX_gnn_type=GATConv_batch_size_by_HADMID=128_loss=5.1474"
# python main.py --use_gpu --val --val_num=1 --gnn_type=GENConv  --val_model_state_dict="task=MIX_gnn_type=GENConv_batch_size_by_HADMID=128_loss=3.5760"

python main.py --use_gpu --val --val_num=2 --gnn_type=GINEConv --val_model_state_dict="task=MIX_gnn_type=GINEConv_batch_size_by_HADMID=128_loss=7.0852"
python main.py --use_gpu --val --val_num=2 --gnn_type=GATConv  --val_model_state_dict="task=MIX_gnn_type=GATConv_batch_size_by_HADMID=128_loss=4.9369"
python main.py --use_gpu --val --val_num=2 --gnn_type=GENConv  --val_model_state_dict="task=MIX_gnn_type=GENConv_batch_size_by_HADMID=128_loss=3.5982"

python main.py --use_gpu --val --val_num=3 --gnn_type=GINEConv --val_model_state_dict="task=MIX_gnn_type=GINEConv_batch_size_by_HADMID=128_loss=7.0301"
python main.py --use_gpu --val --val_num=3 --gnn_type=GATConv  --val_model_state_dict="task=MIX_gnn_type=GATConv_batch_size_by_HADMID=128_loss=4.9566"
python main.py --use_gpu --val --val_num=3 --gnn_type=GENConv  --val_model_state_dict="task=MIX_gnn_type=GENConv_batch_size_by_HADMID=128_loss=3.6829"

python main.py --use_gpu --val --val_num=4 --gnn_type=GINEConv --val_model_state_dict="task=MIX_gnn_type=GINEConv_batch_size_by_HADMID=128_loss=9.2426"
python main.py --use_gpu --val --val_num=4 --gnn_type=GATConv  --val_model_state_dict="task=MIX_gnn_type=GATConv_batch_size_by_HADMID=128_loss=4.9479"
python main.py --use_gpu --val --val_num=4 --gnn_type=GENConv  --val_model_state_dict="task=MIX_gnn_type=GENConv_batch_size_by_HADMID=128_loss=3.5144"

python main.py --use_gpu --val --val_num=5 --gnn_type=GINEConv --val_model_state_dict="task=MIX_gnn_type=GINEConv_batch_size_by_HADMID=128_loss=7.4923"
python main.py --use_gpu --val --val_num=5 --gnn_type=GATConv  --val_model_state_dict="task=MIX_gnn_type=GATConv_batch_size_by_HADMID=128_loss=4.9054"
python main.py --use_gpu --val --val_num=5 --gnn_type=GENConv  --val_model_state_dict="task=MIX_gnn_type=GENConv_batch_size_by_HADMID=128_loss=3.5710"