python -u main.py \
    --mode=test \
    --data_dir=./test_data/foreign-celebrity \
    --batch_size=1 \
    --bfm_path="resources/BFM2009_Model.mat" \
    --ver_uv_index="resources/vertex_uv_ind.npz" \
    --uv_face_mask_path="resources/face_mask.png" \
    --vgg_path="resources/vgg-face.mat" \
    --load_coarse_ckpt=./results/ckpt_coarse_0816/coarse-131000 \
    --load_fine_ckpt=./results/ckpt_fine_dense_photo_d0.001_dn0.001_sn1_suv0.01_bs48_0819/fine-10000 \
    --is_fine_model \
    --output_dir=./results/test_data/foreign-celebrity-0823



