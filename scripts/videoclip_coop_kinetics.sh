# Run 16 shot on GPU 3
CUDA_VISIBLE_DEVICES=7 python hyperparam_search.py videoclip coop kinetics_100 16 &

# Run 8 shot on GPU 2
CUDA_VISIBLE_DEVICES=6 python hyperparam_search.py videoclip coop kinetics_100 8 &

# Run 4 shot on GPU 1
CUDA_VISIBLE_DEVICES=5 python hyperparam_search.py videoclip coop kinetics_100 4 &

# Run 1, 2 shot on GPU 0
CUDA_VISIBLE_DEVICES=4 python hyperparam_search.py videoclip coop kinetics_100 1;
CUDA_VISIBLE_DEVICES=4 python hyperparam_search.py videoclip coop kinetics_100 2;


