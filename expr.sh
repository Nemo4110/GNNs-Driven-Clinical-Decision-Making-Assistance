#!/bin/bash
# following codes are assumed to be executed under conda env "LERS*"

### different batch_size_by_HADMID ###
# python main.py --use_gpu --train --batch_size_by_HADMID=512
# python main.py --use_gpu --train --batch_size_by_HADMID=256
# python main.py --use_gpu --train --batch_size_by_HADMID=128

### different gnn_type ###
#python main.py --use_gpu --train --gnn_type=GINEConv
#python main.py --use_gpu --train --gnn_type=GENConv
#python main.py --use_gpu --train --gnn_type=GATConv

#python main.py --use_gpu --train --gnn_type=GINEConv
#python main.py --use_gpu --train --gnn_type=GENConv
#python main.py --use_gpu --train --gnn_type=GATConv

#python main.py --use_gpu --train --gnn_type=GINEConv
#python main.py --use_gpu --train --gnn_type=GENConv
#python main.py --use_gpu --train --gnn_type=GATConv

#python main.py --use_gpu --train --gnn_type=GINEConv
#python main.py --use_gpu --train --gnn_type=GENConv
#python main.py --use_gpu --train --gnn_type=GATConv

#python main.py --use_gpu --train --gnn_type=GINEConv
#python main.py --use_gpu --train --gnn_type=GENConv
#python main.py --use_gpu --train --gnn_type=GATConv

# python main.py --use_gpu --train --gnn_type=GENConv --batch_size_by_HADMID=256

### testing ###
python main.py --use_gpu --test --test_num=1 --gnn_type=GINEConv --neg_smp_strategy=100 --test_model_state_dict="task=MIX_gnn_type=GINEConv_batch_size_by_HADMID=128_loss=5.7964"
python main.py --use_gpu --test --test_num=1 --gnn_type=GATConv  --neg_smp_strategy=100 --test_model_state_dict="task=MIX_gnn_type=GATConv_batch_size_by_HADMID=128_loss=4.5010"
python main.py --use_gpu --test --test_num=1 --gnn_type=GENConv  --neg_smp_strategy=100 --test_model_state_dict="task=MIX_gnn_type=GENConv_batch_size_by_HADMID=128_loss=3.1443"

python main.py --use_gpu --test --test_num=2 --gnn_type=GINEConv --neg_smp_strategy=100 --test_model_state_dict="task=MIX_gnn_type=GINEConv_batch_size_by_HADMID=128_loss=5.8608"
python main.py --use_gpu --test --test_num=2 --gnn_type=GATConv  --neg_smp_strategy=100 --test_model_state_dict="task=MIX_gnn_type=GATConv_batch_size_by_HADMID=128_loss=4.5607"
python main.py --use_gpu --test --test_num=2 --gnn_type=GENConv  --neg_smp_strategy=100 --test_model_state_dict="task=MIX_gnn_type=GENConv_batch_size_by_HADMID=128_loss=3.2983"

python main.py --use_gpu --test --test_num=3 --gnn_type=GINEConv --neg_smp_strategy=100 --test_model_state_dict="task=MIX_gnn_type=GINEConv_batch_size_by_HADMID=128_loss=5.7413"
python main.py --use_gpu --test --test_num=3 --gnn_type=GATConv  --neg_smp_strategy=100 --test_model_state_dict="task=MIX_gnn_type=GATConv_batch_size_by_HADMID=128_loss=4.6074"
python main.py --use_gpu --test --test_num=3 --gnn_type=GENConv  --neg_smp_strategy=100 --test_model_state_dict="task=MIX_gnn_type=GENConv_batch_size_by_HADMID=128_loss=3.2756"

python main.py --use_gpu --test --test_num=4 --gnn_type=GINEConv --neg_smp_strategy=100 --test_model_state_dict="task=MIX_gnn_type=GINEConv_batch_size_by_HADMID=128_loss=5.9453"
python main.py --use_gpu --test --test_num=4 --gnn_type=GATConv  --neg_smp_strategy=100 --test_model_state_dict="task=MIX_gnn_type=GATConv_batch_size_by_HADMID=128_loss=4.4467"
python main.py --use_gpu --test --test_num=4 --gnn_type=GENConv  --neg_smp_strategy=100 --test_model_state_dict="task=MIX_gnn_type=GENConv_batch_size_by_HADMID=128_loss=3.2079"

python main.py --use_gpu --test --test_num=5 --gnn_type=GINEConv --neg_smp_strategy=100 --test_model_state_dict="task=MIX_gnn_type=GINEConv_batch_size_by_HADMID=128_loss=5.2503"
python main.py --use_gpu --test --test_num=5 --gnn_type=GATConv  --neg_smp_strategy=100 --test_model_state_dict="task=MIX_gnn_type=GATConv_batch_size_by_HADMID=128_loss=4.3887"
python main.py --use_gpu --test --test_num=5 --gnn_type=GENConv  --neg_smp_strategy=100 --test_model_state_dict="task=MIX_gnn_type=GENConv_batch_size_by_HADMID=128_loss=3.2225"

#python main.py --use_gpu --num_gpu=1 --test --gnn_type=GENConv --batch_size_by_HADMID=256 --test_model_state_dict="task=MIX_gnn_type=GENConv_batch_size_by_HADMID=256_loss=4.0512"
