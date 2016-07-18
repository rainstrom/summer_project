import vgg16
import ucf101
import tensorflow as tf
import numpy as np
from load_model import load_from_pickle

root_dir = '/scratch/xiaoyang/UCF101_frames_org2'
train_list = '/home/xiaoyang/action_recognition_exp/dataset_file_examples/train_split1_avi.txt'
test_list = '/home/xiaoyang/action_recognition_exp/dataset_file_examples/val_split1_avi.txt'

learning_rate = 0.001
batch_size = 50
total_steps = 12000; decay_steps = 4000; decay_factor = 0.1
momentum = 0.9
num_segments = 1; num_length = 1
test_inteval = 500; test_iter = 10
save_inteval = 4000
showing_inteval = 20
final_keep_prob_value = 0.2
test_segments = 25
assert batch_size % test_segments == 0

run_training = True
load_parameter_from_tfmodel = True
run_full_test = True

batch_data = tf.placeholder(tf.float32, shape=[batch_size, 224, 224, num_segments*num_length*3], name="data")
batch_label = tf.placeholder(tf.int64, shape=[batch_size], name="label")
keep_prob = tf.placeholder("float")
global_step = tf.Variable(0, name='global_step', trainable=False)

softmax_digits = vgg16.inference(batch_data, keep_prob=keep_prob, train_conv123=False, train_conv45=False, train_fc67=True)
cross_entropy_loss, total_loss = vgg16.loss(softmax_digits, batch_label)
accuracy = vgg16.accuracy(softmax_digits, batch_label)

lr = tf.train.exponential_decay(learning_rate,
                              global_step,
                              decay_steps,
                              decay_factor,
                              staircase=False)
optimizer = tf.train.MomentumOptimizer(lr, momentum)
train_op = optimizer.minimize(total_loss, global_step=global_step)
import pdb; pdb.set_trace()
saver = tf.train.Saver(tf.get_collection("params"))
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=True, gpu_options=gpu_options))
sess.run(tf.initialize_all_variables())

if load_parameter_from_tfmodel:
    print("loading from tfmodel")
    load_from_pickle("VGG_ILSVRC_16_layers.tfmodel", sess, ignore_missing=True)
    print("loaded from tfmodel")
else:
    ckpt = tf.train.get_checkpoint_state("./weights_rgb/")
    assert (ckpt and ckpt.model_checkpoint_path)
    print("loading from cpkt: {}".format(ckpt.model_checkpoint_path))
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("loaded from cpkt")

if run_training:
    data_reader = ucf101.reader(root_dir, train_list, "RGB", batch_size, num_length, num_segments, False)
    test_data_reader = ucf101.reader(root_dir, test_list, "RGB", batch_size, num_length, num_segments, True)
    step = 0
    for i in range(total_steps):
        print("loading data")
        data, label = data_reader.get()
        # import pdb; pdb.set_trace()
        print("running")
        progress = float(step) / total_steps; progress = min(progress * 6, 1.0)
        keep_prob_value = progress * final_keep_prob_value + (1 - progress) * 0.5
        (_, loss_value, acc_value, step, lr_value) = sess.run([train_op, cross_entropy_loss, accuracy, global_step, lr], feed_dict={batch_data:data, batch_label:label, keep_prob: keep_prob_value})
        print("[step %d]: loss, %f; acc, %f; lr, %f; keep, %f" % (step, loss_value, acc_value, lr_value, keep_prob_value))
        if step % test_inteval == 0 or step == 1:
            print("testing ... ")
            all_acc = []
            all_loss = []
            for k in range(test_iter):
                test_data, test_label = test_data_reader.get()
                (loss_value, acc_value) = sess.run([cross_entropy_loss, accuracy], feed_dict={batch_data:test_data, batch_label:test_label, keep_prob: 1.0})
                all_acc.append(acc_value)
                all_loss.append(loss_value)
                print("test iter %d: acc, %f; loss, %f" % (k, acc_value, loss_value))
            print("test result: acc, %f; loss, %f" % (np.mean(all_acc), np.mean(all_loss)))

        if step % save_inteval == 0 or step == 1:
            print("Saving model")
            save_path = saver.save(sess, "./weights_rgb/rgb_vgg16_iter%d.ckpt" % (step))
            print("Model saved in file: %s" % save_path)
        if step >= total_steps:
            break
    print("trainning finished")

if run_full_test:
    assert batch_size % test_segments == 0
    batch_size /= test_segments
    num_segments = test_segments

    full_test_data_reader = ucf101.reader(root_dir, test_list, "RGB", batch_size, num_length, num_segments, True, "FULLTEST")
    full_test_video_num = full_test_data_reader.get_video_num()
    print("full testing ... ... ")
    print("totally test %d videos" % full_test_video_num)
    full_test_all_acc = []
    full_test_final_acc_num = 0
    for i in range(full_test_video_num // batch_size):
        step = i + 1
        test_data, test_label = full_test_data_reader.get()
        (loss_value, acc_value, fc8_value) = sess.run([cross_entropy_loss, accuracy, softmax_digits], feed_dict={batch_data:test_data, batch_label:test_label, keep_prob: 1.0})
        print("test iter %d: acc, %f; loss, %f" % (step, acc_value, loss_value))
        full_test_all_acc.append(acc_value)
        for k in range(batch_size):
            fc8_value_1 = fc8_value[k*num_segments:(k+1)*num_segments, :]
            test_label_1 = test_label[k*num_segments:(k+1)*num_segments]
            assert all([test_label_1[0]==lb for lb in test_label_1])
            test_label_1 = test_label_1[0]
            fc8_value_1_mean = np.mean(fc8_value_1, axis=0)
            final_pd = np.argmax(fc8_value_1_mean)
            if int(final_pd) == int(test_label_1):
                full_test_final_acc_num += 1
        print("step %d: 1 frame acc is %f" % (step, np.mean(full_test_all_acc)))
        print("step %d: %d frame acc is %f" % (step, num_segments, float(full_test_final_acc_num) / float(step * batch_size)))

    print("final 1 frame acc is %f" % np.mean(full_test_all_acc))
    print("final %d frame acc is %f" % (num_segments, float(full_test_final_acc_num) / float(full_test_video_num)))
