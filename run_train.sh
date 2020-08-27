# training script for coarse model
python -u main.py \
    --mode=train \
    --landmark3d_weight=1.0 \
    --landmark2d_weight=0.5 \
    --photo_weight=1.0 \
    --reg_shape_weight=3.0 \
    --reg_exp_weight=10.0 \
    --reg_tex_weight=1.0 \
    --id_weight=0.8 \
    --data_dir=./train_data \
    --batch_size=16 \
    --epoch=50 \
    --num_threads=10 \
    --bfm_path="resources/BFM2009_Model.mat" \
    --ver_uv_index="resources/vertex_uv_ind.npz" \
    --vgg_path="resources/vgg-face.mat" \
    --uv_face_mask_path="resources/face_mask.png" \
    --learning_rate=0.00001 \
    --lr_decay_step=2000 \
    --lr_decay_rate=0.98 \
    --min_learning_rate=0.000001 \
    --summary_dir=./results/summary_coarse \
    --coarse_ckpt=./results/ckpt_coarse \
    --step=100000 \
    --save_step=1000 \
    --log_step=100 \
    --obj_step=500

# training script for fine model
#python -u main.py \
#    --mode=train \
#    --landmark3d_weight=0.0 \
#    --landmark2d_weight=0.0 \
#    --photo_weight=1.0 \
#    --reg_shape_weight=0.0 \
#    --reg_exp_weight=0.0 \
#    --reg_tex_weight=0.0 \
#    --id_weight=0.0 \
#    --smooth_weight=1.0 \
#    --smooth_uv_weight=0.01 \
#    --smooth_normal_weight=1.0000 \
#    --disp_weight=0.001 \
#    --disp_normal_weight=0.001 \
#    --data_dir=./train_data \
#    --batch_size=48 \
#    --epoch=5 \
#    --num_threads=10 \
#    --bfm_path="resources/BFM2009_Model.mat" \
#    --ver_uv_index="resources/vertex_uv_ind.npz" \
#    --uv_face_mask_path="resources/face_mask.png" \
#    --vgg_path="resources/vgg-face.mat" \
#    --learning_rate=0.00001 \
#    --is_fine_model \
#    --lr_decay_step=5000 \
#    --lr_decay_rate=0.99 \
#    --min_learning_rate=0.0000001 \
#    --summary_dir=./results/summary_fine \
#    --fine_ckpt=./results/ckpt_fine \
#    --step=10001 \
#    --save_step=1000 \
#    --log_step=100 \
#    --obj_step=500
#
