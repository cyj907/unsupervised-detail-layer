import tensorflow as tf
import os
import cv2
import glob
import numpy as np
from PIL import Image
import face_model
import losses
from tools.basis_utils import load_3dmm_basis
from tools.data_utils import DataSet
from tools.ply import write_ply
from tools.misc import MaskCreator


def compute_losses(image_load,
        landmark3d_load,
        landmark2d_load,
        seg_load,
        ret_results,
        basis3dmm,
        args):

    tot_loss = float(0.)

    if args.landmark3d_weight > 0:
        if args.is_fine_model is False:
            landmark3d_loss = losses.weighted_landmark_loss(landmark3d_load, ret_results['coarse']['ver']['landmark'])
        tot_loss = tot_loss + landmark3d_loss * args.landmark3d_weight
        tf.summary.scalar('landmark3d', landmark3d_loss)

    if args.landmark2d_weight > 0:
        if args.is_fine_model is False:
            # contour only
            landmark2d_loss = losses.landmark2d_loss(
                    landmark2d_load[:,:18,:],
                    ret_results['coarse']['ver']['proj_xy'],
                    ret_results['coarse']['ver']['contour_mask'],
                    ret_results['coarse']['ver']['normal'],
                    basis3dmm['keypoints'][:18])
            tf.summary.scalar('landmark2d', landmark2d_loss)
            tot_loss = tot_loss + landmark2d_loss * args.landmark2d_weight

    if args.photo_weight > 0:
        if args.is_fine_model:
            photo_mask = MaskCreator.create_fine_photo_loss_mask_from_seg(seg_load)
            photo_mask = tf.clip_by_value(photo_mask * ret_results['fine']['screen']['mask_face'], 0, 1)
            photo_loss = losses.photo_loss(ret_results['fine']['screen']['inputs'], ret_results['fine']['screen']['rgb'], photo_mask)

            tf.summary.image('render_fine_seg', tf.cast(photo_mask * 255, tf.uint8), max_outputs=1)
            tf.summary.image('render_coarse_photo', tf.cast(ret_results['coarse']['screen']['rgb'], tf.uint8), max_outputs=1)
            tf.summary.image('render_fine_photo', tf.cast(ret_results['fine']['screen']['rgb'], tf.uint8), max_outputs=1)
        else:
            photo_mask = MaskCreator.create_coarse_photo_loss_mask_from_seg(seg_load)
            photo_mask = tf.clip_by_value(photo_mask * ret_results['coarse']['screen']['photo_mask'], 0, 1)
            photo_loss = losses.photo_loss(image_load, ret_results['coarse']['screen']['rgb'], photo_mask)
            tf.summary.image('render_coarse_photo', tf.cast(ret_results['coarse']['screen']['rgb'], tf.uint8), max_outputs=1)
            tf.summary.image('render_coarse_seg', tf.cast(photo_mask * 255, tf.uint8), max_outputs=1)
        tot_loss = tot_loss + photo_loss * args.photo_weight
        tf.summary.scalar('photo', photo_loss)
        tf.summary.image('input_photo', tf.cast(image_load, tf.uint8), max_outputs=1)

    reg_loss = float(0.)
    if args.reg_shape_weight > 0:
        reg_shape_loss = losses.reg_loss(ret_results['coarse']['para']['shape'], 'shape')
        reg_loss = reg_loss + reg_shape_loss * args.reg_shape_weight
        tf.summary.scalar('reg_shape', reg_shape_loss)

    if args.reg_exp_weight > 0:
        reg_exp_loss = losses.reg_loss(ret_results['coarse']['para']['exp'], 'exp')
        reg_loss = reg_loss + reg_exp_loss * args.reg_exp_weight
        tf.summary.scalar('reg_exp', reg_exp_loss)

    if args.reg_tex_weight > 0:
        reg_tex_loss = losses.reg_loss(ret_results['coarse']['para']['tex'], 'tex')
        reg_loss = reg_loss + reg_tex_loss * args.reg_tex_weight
        tf.summary.scalar('reg_tex', reg_tex_loss)
    tot_loss = tot_loss + reg_loss

    if args.id_weight > 0:
        if args.is_fine_model:
            id_loss = losses.id_loss(image_load, ret_results['fine']['screen']['rgb'], args.vgg_path)
        else:
            id_loss = losses.id_loss(image_load, ret_results['coarse']['screen']['rgb'], args.vgg_path)
        tf.summary.scalar('id_loss', id_loss)
        tot_loss = tot_loss + id_loss * args.id_weight

    if args.is_fine_model:

        # displacement depth loss
        if args.disp_weight:
            ver_mask = ret_results['fine']['ver']['mask_face']
            ver_mask_wo_eyebrow_nose = ret_results['fine']['ver']['mask_disp']
            disp_loss = losses.disp_loss(ret_results['fine']['ver']['disp_depth'], ver_mask, 'disp')
            render_disp_loss = losses.disp_loss(ret_results['fine']['screen']['depth']- ret_results['coarse']['screen']['depth'], ret_results['fine']['screen']['mask_face'], 'render_disp')

            disp_eyebrow_nose_loss = losses.disp_loss(ret_results['fine']['ver']['disp_depth'], 1 - ver_mask_wo_eyebrow_nose, 'disp_eyebrow_nose')
            tf.summary.scalar('disp_loss', disp_loss)
            tf.summary.scalar('disp_eyebrow_nose_loss', disp_eyebrow_nose_loss)
            tf.summary.scalar('render_disp_loss', render_disp_loss)
            tot_loss = tot_loss + (disp_loss + render_disp_loss * 2) * args.disp_weight + disp_eyebrow_nose_loss * args.disp_weight * 2

        if args.disp_normal_weight:
            disp_normal_loss = losses.disp_loss(ret_results['fine']['ver']['normal'] - ret_results['fine']['ver']['coarse_normal'], ret_results['fine']['ver']['mask_face'],'normal')
            tf.summary.scalar('disp_normal_loss', disp_normal_loss)
            tot_loss = tot_loss + disp_normal_loss * args.disp_normal_weight

        if args.smooth_weight:
            smooth_loss = losses.smooth_loss(
                    ret_results['fine']['ver']['xyz'],
                    ret_results['fine']['ver']['adj'],
                    0.2,
                    'disp')
            tf.summary.scalar('smooth_loss', smooth_loss)
            tot_loss = tot_loss + smooth_loss * args.smooth_weight

        if args.smooth_normal_weight:
            smooth_normal_loss = losses.smooth_loss(
                    ret_results['fine']['ver']['normal'],
                    ret_results['fine']['ver']['adj'],
                    1e-3,
                    'normal')

            smooth_tri_normal_loss = losses.smooth_tri_loss(
                    ret_results['fine']['tri']['normal'],
                    ret_results['fine']['tri']['adj'],
                    1e-4,
                    'normal')
            tf.summary.scalar('smooth_normal_loss', smooth_normal_loss)
            tf.summary.scalar('smooth_tri_normal_loss', smooth_tri_normal_loss)
            tot_loss = tot_loss + smooth_normal_loss * args.smooth_normal_weight + \
                    smooth_tri_normal_loss * args.smooth_normal_weight

        if args.smooth_uv_weight:
            smooth_uv_loss = losses.smooth_uv_loss(
                    ret_results['fine']['uv']['disp_depth'],
                    'disp'
                    )
            tf.summary.scalar('smooth_uv_loss', smooth_uv_loss)

            #smooth_duv_loss = losses.smooth_duv_loss(
            #        ret_results['fine']['uv']['disp_depth'],
            #        'disp'
            #        )
            #tf.summary.scalar('smooth_duv_loss', smooth_duv_loss)

            smooth_slope_uv_loss = losses.smooth_slope_uv_loss(
                    ret_results['fine']['uv']['disp_depth'],
                    'disp'
                    )
            tf.summary.scalar('smooth_slope_uv_loss', smooth_slope_uv_loss)

            tot_loss = tot_loss + (smooth_uv_loss + smooth_slope_uv_loss) * args.smooth_uv_weight

    tf.summary.scalar('tot_loss', tot_loss)

    return tot_loss


def train(args):
    print(args)
    options = {
        "aug": True,
        "color": True,
        "scale": True,
        "flip": True,
    }

    # data loader
    image_load, landmark2d_load, landmark3d_load, seg_load, glassframe_load = \
            DataSet.load(
                    args.data_dir, args.batch_size, options
                    )

    # train basis and other information (e.g., parameters)
    basis3dmm = load_3dmm_basis(args.bfm_path, args.ver_uv_index, args.uv_face_mask_path, './resources/wo_eyebrow_mask.npy', './resources/wo_nose_mask.npy')

    print('building model')
    ret_results = face_model.build_model(image_load, args.vgg_path, basis3dmm, trainable=True, is_fine_model=args.is_fine_model)

    # losses
    print('computing losses')
    tot_loss = compute_losses(image_load, landmark3d_load, landmark2d_load, seg_load, ret_results, basis3dmm, args)

    # training step and learning rate
    if args.is_fine_model:
        global_step = tf.Variable(0, dtype=tf.int32, name='global_step_fine', trainable=False)
        learning_rate = tf.maximum(tf.train.exponential_decay(args.learning_rate,
                                global_step,
                                args.lr_decay_step,
                                args.lr_decay_rate),
                            args.min_learning_rate)
    else:
        global_step = tf.Variable(0, dtype=tf.int32, name='global_step_coarse', trainable=False)
        learning_rate = tf.maximum(tf.train.exponential_decay(args.learning_rate,
                                global_step,
                                args.lr_decay_step,
                                args.lr_decay_rate),
                            args.min_learning_rate)

    # trainable variables
    train_vars = tf.trainable_variables()
    coarse_vars = []
    fine_vars = []
    for var in train_vars:
        if var.name.startswith('CoarseModel'):
            coarse_vars.append(var)
        if var.name.startswith('FineModel'):
            fine_vars.append(var)

    # optimizer
    optim = tf.train.AdamOptimizer(learning_rate)
    gvs = optim.compute_gradients(tot_loss)
    capped_gvs = []

    if args.is_fine_model:
        for grad, var in gvs:
            if grad is not None and var.name.startswith('FineModel'):
                capped_gvs.append((tf.clip_by_value(grad, -1, 1), var))
    else:
        for grad, var in gvs:
            if grad is not None and var.name.startswith('CoarseModel'):
                capped_gvs.append((tf.clip_by_value(grad, -1, 1), var))

    train_op = optim.apply_gradients(capped_gvs, global_step=global_step)

    if args.is_fine_model:
        saver_coarse = tf.train.Saver(coarse_vars)
        saver_fine = tf.train.Saver(fine_vars + [global_step])
    else:
        saver_coarse = tf.train.Saver(coarse_vars + [global_step])

    # summary writer
    summary_writer = tf.summary.FileWriter(args.summary_dir)
    summary_op = tf.summary.merge_all()

    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1.0

    print('before sess run')

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        # load checkpoint
        if args.resume:
            if args.is_fine_model:
                saver_coarse.restore(sess, args.load_coarse_ckpt) # coarse
                saver_fine.restore(sess, args.load_fine_ckpt) # fine
            else:
                saver_coarse.restore(sess, args.load_coarse_ckpt) # coarse
        else:
            if args.is_fine_model:
                saver_coarse.restore(sess, args.load_coarse_ckpt) # coarse model

        cur_step = sess.run(global_step)

        for _ in range(args.step):

            # save results
            if cur_step % args.save_step == 0:
                if args.is_fine_model:
                    saver_fine.save(
                            sess,
                            os.path.join(args.fine_ckpt, 'fine'),
                            global_step=cur_step,
                            write_meta_graph=False
                            )
                else:
                    saver_coarse.save(
                            sess,
                            os.path.join(args.coarse_ckpt, 'coarse'),
                            global_step=cur_step,
                            write_meta_graph=False
                            )

            # save summary
            if cur_step % args.log_step == 0 or cur_step == 1:
                summary = sess.run(summary_op)
                summary_writer.add_summary(summary, cur_step)

            if cur_step % args.obj_step == 0:
                if args.is_fine_model:
                    outputs, input_images = sess.run([ret_results, image_load])
                    ver_xyz_fine = outputs['fine']['ver']['xyz'][0]
                    ver_rgb_fine = outputs['fine']['ver']['rgb'][0]
                    ver_xyz_coarse = outputs['coarse']['ver']['xyz'][0]
                    ver_rgb_coarse = outputs['coarse']['ver']['rgb'][0]
                    image = input_images[0]
                    render_coarse = outputs['coarse']['screen']['rgb'][0]
                    render_fine = outputs['fine']['screen']['rgb'][0]
                    fine_tri = outputs['fine']['tri']['index']
                    uv_rgb = outputs['fine']['uv']['rgb'][0]
                    uv_input = outputs['fine']['uv']['input'][0]

                    write_ply(
                              os.path.join(args.summary_dir, '%d.ply' % cur_step),
                              ver_xyz_coarse,
                              basis3dmm['tri'],
                              ver_rgb_coarse.astype(np.uint8),
                              True)

                    write_ply(
                              os.path.join(args.summary_dir, '%d_fine.ply' % cur_step),
                              ver_xyz_fine,
                              fine_tri,
                              ver_rgb_fine.astype(np.uint8),
                              True)

                    cv2.imwrite(os.path.join(args.summary_dir, '%d_input.png' % cur_step), image[:,:,::-1])
                    cv2.imwrite(os.path.join(args.summary_dir, '%d_coarse.png' % cur_step), render_coarse[:,:,::-1])
                    cv2.imwrite(os.path.join(args.summary_dir, '%d_fine.png' % cur_step), render_fine[:,:,::-1])
                    cv2.imwrite(os.path.join(args.summary_dir, '%d_uv_rgb.png' % cur_step), uv_rgb[:,:,::-1])
                    cv2.imwrite(os.path.join(args.summary_dir, '%d_uv_input.png' % cur_step), uv_input[:,:,::-1])

                else:
                    outputs, input_images = sess.run([ret_results, image_load])
                    ver_xyz_coarse = outputs['coarse']['ver']['xyz'][0]
                    ver_rgb_coarse = outputs['coarse']['ver']['rgb'][0]
                    image = input_images[0]
                    render_coarse = outputs['coarse']['screen']['rgb'][0]

                    write_ply(
                              os.path.join(args.summary_dir, '%d.ply' % cur_step),
                              ver_xyz_coarse,
                              basis3dmm['tri'],
                              ver_rgb_coarse.astype(np.uint8),
                              True)

                    cv2.imwrite(os.path.join(args.summary_dir, '%d_input.png' % cur_step), image[:,:,::-1])
                    cv2.imwrite(os.path.join(args.summary_dir, '%d_coarse.png' % cur_step), render_coarse[:,:,::-1])

            _, cur_step = sess.run([train_op, global_step])
            print(cur_step)


def test(args):

    from preprocess import face_detection, face_cropping, landmark_detection

    # input image

    image_paths = glob.glob(os.path.join(args.data_dir, '*.jpg')) \
            + glob.glob(os.path.join(args.data_dir, '*.png')) \
            + glob.glob(os.path.join(args.data_dir, '.jpeg'))
    image_paths.sort()
    assert(len(image_paths) > 0)

    # inference coarse / fine

    basis3dmm = load_3dmm_basis(
            args.bfm_path,
            args.ver_uv_index,
            args.uv_face_mask_path,
            './resources/wo_eyebrow_mask.npy',
            './resources/wo_nose_mask.npy')

    # load image by batch
    times = len(image_paths) // args.batch_size
    if times * args.batch_size < len(image_paths):
        times += 1

    N = len(image_paths)
    for i_time in range(times):
        i_begin = i_time * args.batch_size
        i_end = min((i_time+1)*args.batch_size, N)
        print(i_begin, i_end)

        image_list = []
        for i_img in range(i_begin, i_end):
            img_path = image_paths[i_img]
            img = np.asarray(Image.open(img_path), np.float32)
            image_list.append(img)

        try:
            print('face detection')
            pb_path = './preprocess/frozen_models/mtcnn_model.pb'
            image_list, bbox_list, point5_list = face_detection.detect_with_MTCNN(image_list, pb_path)

            print('start detect 86pt 3D lmk')
            tf.reset_default_graph()
            pb_path = './preprocess/frozen_models/lmk86_model.pb'
            landmark_list = landmark_detection.detect_lmk86(image_list, bbox_list, point5_list, pb_path)

            print('start crop by 3D lmk')
            crop_image_list = []
            for img, lmk in zip(image_list, landmark_list):
                crop_img, _ = face_cropping.process_image_landmark(img, lmk)
                crop_image_list.append(crop_img)
            crop_image_arr = np.array(crop_image_list, np.float32)
        except Exception as e:
            print(e)
            exit()

        tf.reset_default_graph()
        with tf.Session() as sess:

            image_load = tf.placeholder(tf.float32, [i_end-i_begin, 300, 300, 3], name='images')
            ret_results = face_model.build_model(image_load, args.vgg_path, basis3dmm, trainable=True, is_fine_model=args.is_fine_model)

            sess.run(tf.global_variables_initializer())

            # checkpoints loading
            train_vars = tf.trainable_variables()
            coarse_vars = []
            fine_vars = []
            for var in train_vars:
                if var.name.startswith('CoarseModel'):
                    coarse_vars.append(var)
                if var.name.startswith('FineModel'):
                    fine_vars.append(var)

            saver_coarse = tf.train.Saver(coarse_vars)
            saver_fine = tf.train.Saver(fine_vars)
            saver_coarse.restore(sess, args.load_coarse_ckpt)
            saver_fine.restore(sess, args.load_fine_ckpt)

            outputs = sess.run(ret_results, {image_load: crop_image_arr})

            # save results to ply, low/fine/coarse
            ver_xyz_coarse_results = outputs['coarse']['ver']['xyz']
            ver_rgb_coarse_results = outputs['coarse']['ver']['rgb']
            ver_rgb_fine_results = outputs['fine']['ver']['rgb']
            ver_xyz_fine_results = outputs['fine']['ver']['xyz']
            fine_tri = outputs['fine']['tri']['index']

            if os.path.exists(args.output_dir) is False:
                os.makedirs(args.output_dir)

            for i in range(i_end - i_begin):
                image_path = image_paths[i_begin + i]
                print(image_path)
                name = image_path.split('/')[-1].split('.')[0]
                Image.fromarray(crop_image_arr[i].astype(np.uint8)).save(os.path.join(args.output_dir, '%s.png' % name))
                write_ply(
                          os.path.join(args.output_dir, '%s.ply' % name),
                          ver_xyz_coarse_results[i],
                          basis3dmm['tri'],
                          ver_rgb_coarse_results[i].astype(np.uint8),
                          True)

                write_ply(
                          os.path.join(args.output_dir, '%s_fine.ply' % name),
                          ver_xyz_fine_results[i],
                          fine_tri,
                          ver_rgb_fine_results[i].astype(np.uint8),
                          True)

