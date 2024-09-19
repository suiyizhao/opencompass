source ~/anaconda3/etc/profile.d/conda.sh
conda activate opencompass

# export http_proxy=http://zhaosuiyi:GEESg7ZGUyKLyzuDOCTQWLPLJEELKnxhbgqbj6JkSZ6uRFfABIQuTL3irp52@10.1.20.57:23128/
# export https_proxy=http://zhaosuiyi:GEESg7ZGUyKLyzuDOCTQWLPLJEELKnxhbgqbj6JkSZ6uRFfABIQuTL3irp52@10.1.20.57:23128/
# export HTTP_PROXY=http://zhaosuiyi:GEESg7ZGUyKLyzuDOCTQWLPLJEELKnxhbgqbj6JkSZ6uRFfABIQuTL3irp52@10.1.20.57:23128/
# export HTTPS_PROXY=http://zhaosuiyi:GEESg7ZGUyKLyzuDOCTQWLPLJEELKnxhbgqbj6JkSZ6uRFfABIQuTL3irp52@10.1.20.57:23128/

config=eval_CT20b-0912-ckpt-hf21907__internlm2-5_train_tempe0

cd /mnt/petrelfs/zhaosuiyi/projects/opencompass
python run.py ./configs/$config.py

conda deactivate