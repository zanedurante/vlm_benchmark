# TODO: Need to add conda environment activation for each test
conda activate vlm_benchmark;
python -m tests.test_CLIP;
conda activate videoclip;
python -m tests.test_videoclip;
conda activate frozen;
python -m tests.test_frozen;