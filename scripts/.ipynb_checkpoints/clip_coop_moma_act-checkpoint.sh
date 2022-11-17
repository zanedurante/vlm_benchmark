# Run 16 shot on GPU 3
CUDA_VISIBLE_DEVICES=7 python hyperparam_search.py clip coop moma_act 16 &

# Run 8 shot on GPU 2
CUDA_VISIBLE_DEVICES=6 python hyperparam_search.py clip coop moma_act 8 &

# Run 4 shot on GPU 1
CUDA_VISIBLE_DEVICES=5 python hyperparam_search.py clip coop moma_act 4 &

# Run 1, 2 shot on GPU 0
CUDA_VISIBLE_DEVICES=4 python hyperparam_search.py clip coop moma_act 1;
CUDA_VISIBLE_DEVICES=4 python hyperparam_search.py clip coop moma_act 2;


