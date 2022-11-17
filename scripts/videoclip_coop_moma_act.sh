# Run 16 shot on GPU 3
CUDA_VISIBLE_DEVICES=0 python hyperparam_search.py videoclip coop moma_act 16 2 &

# Run 8 shot on GPU 2
CUDA_VISIBLE_DEVICES=1 python hyperparam_search.py videoclip coop moma_act 8 2 &

# Run 4 shot on GPU 1
CUDA_VISIBLE_DEVICES=2 python hyperparam_search.py videoclip coop moma_act 4 2 &

# Run 1, 2 shot on GPU 0
CUDA_VISIBLE_DEVICES=3 python hyperparam_search.py videoclip coop moma_act 1 2;
CUDA_VISIBLE_DEVICES=3 python hyperparam_search.py videoclip coop moma_act 2 2;


