# only structure feature

python GENI/geni_batch_train.py \
    --dataset FB15k_rel \
    --data_path datasets/tot_data.pk \
    --gpu 2 \
    --spm \
    --save-path geni_spm_checkpoint.pt