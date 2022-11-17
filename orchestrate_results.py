import multiprocessing
from multiprocessing import Process, Lock
from refactored_hyperparam_search import get_results
import time
import torch 

NUM_GPUS = 8


def f(gpu_locks, args):
    while True: # Infinite loop until it finds a GPU break out 
        for gpu_idx, gpu_lock in enumerate(gpu_locks):
            acq = gpu_lock.acquire(block=False)
            if acq:
                import os
                os.environ['CUDA_VISIBLE_DEVICES']=str(gpu_idx)
                torch.cuda.set_device(gpu_idx)
                print('Using gpu:', gpu_idx, 'with args', args)
                get_results(**args)
                print('Returning gpu:', gpu_idx)
                gpu_lock.release()
                return
        time.sleep(100) # Wait 100 seconds before looping again to conserve CPU resources

"""
VLM_ARG, CLASSIFIER_ARG, dataset_name=["smsm", "kinetics_100"], 
                num_shots = [1, 2, 4, 8, 16], num_episodes = [4]
"""
            
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    gpu_locks = [Lock() for _ in range(NUM_GPUS)]
    classifier_arg = "cona"
    vlms = ["videoclip", "clip", "miles"] # Order slowest to fastest
    num_episodes = [2, 4, 4] # Corresponds to ^
    datasets = ["kinetics_100", "moma_act"] # Order largest to smallest
    num_shots = [16, 8, 4, 2, 1] # Order largest to smallest
    
    default_args_to_run = [] # Running with CoNa
    for vlm, n_episodes in zip(vlms, num_episodes):
        for dataset in datasets:
            for n_shots in num_shots:
                default_args_to_run.append({
                    "VLM_ARG":vlm,
                    "CLASSIFIER_ARG":classifier_arg,
                    "dataset_name":[dataset],
                    "num_shots":[n_shots],
                    "num_episodes":[n_episodes],
                })
    processes = []
    for args in default_args_to_run:
        processes.append(Process(target=f, args=(gpu_locks, args)))
        processes[-1].start()
        time.sleep(1) # Allow enough time for first processes to get GPUs first
    for process in processes:
        process.join()