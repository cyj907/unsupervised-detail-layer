import tensorflow as tf
import controller
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, help="train / test")

parser.add_argument("--data_dir", help="path to folder containing tf records[train] / input images[test]")
parser.add_argument("--output_dir", help="path to folder with test results")
parser.add_argument("--batch_size", type=int, help="batch size")
parser.add_argument("--epoch", type=int, help="epoch")
parser.add_argument("--num_threads", type=int, help="number of threads to load data")
parser.add_argument("--bfm_path", type=str, help="path to bfm bases")
parser.add_argument("--ver_uv_index", type=str, help="path to vertex to uv indices")
parser.add_argument("--uv_face_mask_path", type=str, help="path to uv face mask")
parser.add_argument("--vgg_path", type=str, help="path to vgg face checkpoints")

parser.add_argument("--learning_rate", type=float, help="learning rate")
parser.add_argument("--lr_decay_step", type=int, help="learning rate decay step")
parser.add_argument("--lr_decay_rate", type=float, help="learning rate decay rate")
parser.add_argument("--min_learning_rate", type=float, help="minimum learning rate")

parser.add_argument("--is_fine_model", action="store_true", help="whether trained as fine model")

parser.add_argument("--resume", action="store_true", help="whether resume training")
parser.add_argument("--summary_dir", type=str, help="summary directory")
parser.add_argument("--coarse_ckpt", type=str, help="path to checkpoints directory for coarse model")
parser.add_argument("--load_coarse_ckpt", type=str, help="path to checkpoints directory for coarse model")
parser.add_argument("--fine_ckpt", type=str, help="path to checkpoints directory for fine model")
parser.add_argument("--load_fine_ckpt", type=str, help="path to checkpoints directory for fine model")

parser.add_argument("--step", type=int, help="training steps")
parser.add_argument("--save_step", type=int, help="step to save model")
parser.add_argument("--log_step", type=int, help="step to log model")
parser.add_argument("--obj_step", type=int, help="step to save 3d files")

parser.add_argument("--landmark3d_weight", type=float, help="landmark 3d weight")
parser.add_argument("--landmark2d_weight", type=float, help="landmark 2d weight")
parser.add_argument("--photo_weight", type=float, help="photo weight")
parser.add_argument("--id_weight", type=float, help="id weight")
parser.add_argument("--reg_shape_weight", type=float, help="reg shape weight")
parser.add_argument("--reg_exp_weight", type=float, help="reg exp weight")
parser.add_argument("--reg_tex_weight", type=float, help="reg tex weight")
parser.add_argument("--disp_weight", type=float, help="displacement weight")
parser.add_argument("--disp_normal_weight", type=float, help="displacement normal weight")
parser.add_argument("--smooth_weight", type=float, help="smoothing weight")
parser.add_argument("--smooth_normal_weight", type=float, help="smoothing normal weight")
parser.add_argument("--smooth_uv_weight", type=float, help="smoothing uv weight")
args = parser.parse_args()




if __name__ == '__main__':
    if args.mode == 'train':
        controller.train(args)
    elif args.mode == 'test':
        controller.test(args)
    else:
        raise NotImplementedError(("The specified mode is not valid: %s\nTry 'train' or 'test' instead." % args.mode))

