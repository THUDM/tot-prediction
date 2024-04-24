# python inference.py \
#     --dataset FB15k_rel \
#     --data_path datasets/tot_data.pk \
#     --gpu 3 \
#     --scale \
#     --output_dir /home/zhangfanjin/ssj/RGTN-NIE/results/RB15k_rel_GENI_inference \
#     --cross_id 0 \
#     --model_path /home/zhangfanjin/ssj/RGTN-NIE/results/FB15k_rel_GENI/0_geni_spm_checkpoint.pt


python inference.py \
    --dataset FB15k_rel_two \
    --data_path datasets/tot_data.pk \
    --gpu 3 \
    --scale \
    --output_dir /home/zhangfanjin/ssj/RGTN-NIE/results/RB15k_rel_GENI_inference \
    --cross_id 0 \
    --model_path /home/zhangfanjin/ssj/RGTN-NIE/results/FB15k_rel_GENI/0_geni_spm_checkpoint.pt