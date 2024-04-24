# only structure feature

python GENI/geni_train.py \
    --dataset FB15k_rel \
    --data_path datasets/tot_data.pk \
    --gpu 3 \
    --spm \
    --save-path geni_spm_checkpoint.pt \
    --scale