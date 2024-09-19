source /mnt/petrelfs/liuyanxing/.bashrc
# open proxy
export http_proxy=http://liuyanxing:r7YHaOv2VZ3ULDSqlboDch9VAwhpabDRWPe9A5MDhjlkKnBjAdfKtcNgsHna@10.1.20.51:23128/
export https_proxy=http://liuyanxing:r7YHaOv2VZ3ULDSqlboDch9VAwhpabDRWPe9A5MDhjlkKnBjAdfKtcNgsHna@10.1.20.51:23128/
export HTTP_PROXY=http://liuyanxing:r7YHaOv2VZ3ULDSqlboDch9VAwhpabDRWPe9A5MDhjlkKnBjAdfKtcNgsHna@10.1.20.51:23128/
export HTTPS_PROXY=http://liuyanxing:r7YHaOv2VZ3ULDSqlboDch9VAwhpabDRWPe9A5MDhjlkKnBjAdfKtcNgsHna@10.1.21.50:23128/
conda activate opencompass

python /mnt/petrelfs/liuyanxing/projects/opencompass/run.py configs/opencompass/eval_internlm2_7b_chat_comac_subjective_0711.py --hf-num-gpus 4 --max-num-workers 4 --max-workers-per-gpu 1 -r
