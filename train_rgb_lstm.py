import vgg16_lstm
import ucf101
import tensorflow as tf
import numpy as np
from load_model import load_from_pickle
import math
import signal
import sys
import summary

root_dir = '/scratch/xiaoyang/UCF101_frames_org2'
train_list = '/home/xiaoyang/action_recognition_exp/dataset_file_examples/train_split1_avi.txt'
test_list = '/home/xiaoyang/action_recognition_exp/dataset_file_examples/val_split1_avi.txt'

learning_rate = 0.01
batch_size = 25 #
total_steps = 30000; decay_steps = 10000; decay_factor = 0.1
momentum = 0.9
num_segments = 25; num_length = 1
test_inteval = 1000; test_iter = 125
save_inteval = 3000
showing_inteval = 20
cnn_keep_prob_value = 1.0
lstm_keep_prob_value = 0.4

run_training = True
run_full_test = True
load_parameter_from_tfmodel = False

batch_data = tf.placeholder(tf.float32, shape=[num_segments, batch_size, 224, 224, num_length*3], name="data")
batch_label = tf.placeholder(tf.int64, shape=[batch_size], name="label")
lstm_keep_prob = tf.placeholder("float")
global_step = tf.Variable(0, name='global_step', trainable=False)

softmax_digits = vgg16_lstm.inference(batch_data, batch_size, num_segments, lstm_keep_prob, 1.0)
cross_entropy_loss, total_loss = vgg16_lstm.loss(softmax_digits, batch_label, num_segments)
accuracy = vgg16_lstm.accuracy(softmax_digits, batch_label, num_segments)

# lr = learning_rate
lr = tf.train.exponential_decay(learning_rate,
                              global_step,
                              decay_steps,
                              decay_factor,
                              staircase=False)
# optimizer = tf.train.MomentumOptimizer(lr, momentum)
optimizer = tf.train.AdamOptimizer(lr)
gvs = optimizer.compute_gradients(total_loss)
capped_gvs = [(tf.clip_by_norm(grad, 3.0), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

saver = tf.train.Saver(tf.all_variables())
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.initialize_all_variables())

tf.scalar_summary("cross_entropy_loss", cross_entropy_loss)
tf.scalar_summary("total_loss", total_loss)
tf.scalar_summary("learning_rate", lr)
tf.scalar_summary("accuracy", accuracy)
for grad, var in gvs:
    summary.variable_summaries(grad, var.name + "_grad")
    summary.variable_summaries(var, var.name + "_var")
summary_writer = tf.train.SummaryWriter("summary_rgb_lstm", sess.graph)
merged_summaries = tf.merge_all_summaries()
do_summary = True

if load_parameter_from_tfmodel:
    print("loading from tfmodel")
    load_from_pickle("VGG_ILSVRC_16_layers.tfmodel", sess, ignore_missing=True)
    print("loaded from tfmodel")
else:
    ckpt = tf.train.get_checkpoint_state("./weights_rgb_lstm/")
    assert (ckpt and ckpt.model_checkpoint_path)
    print("loading from cpkt: {}".format(ckpt.model_checkpoint_path))
    #conv_saver = tf.train.Saver({var.name[5:-2]:var for var in tf.get_collection("params") if var.name.startswith("conv")})
    #conv_saver.restore(sess, ckpt.model_checkpoint_path)
    #saver1 = tf.train.Saver(tf.get_collection("params"))
    #saver1.restore(sess, ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("loaded from cpkt")

start_step = sess.run(global_step)

def analysis_test_result(num_segments, test_iter, loss_values, acc_values, fc8_values, label_values):
    assert len(loss_values) == test_iter
    assert len(acc_values) == test_iter
    total_video_num = float(fc8_values[0].shape[0]) / num_segments * test_iter
    mean_acc_num = 0
    final_acc_num = 0
    for (fc8_value, test_label) in zip(fc8_values, label_values):
        for k in range(fc8_values[0].shape[0] / num_segments):
            fc8_value_1 = fc8_value[k*num_segments:(k+1)*num_segments, :]
            fc8_value_1 = np.mean(fc8_value_1, axis=0)
            fc8_value_f = fc8_value[(k+1)*num_segments-1, :]
            test_label_1 = test_label[k]
            final_pd_1 = np.argmax(fc8_value_1)
            final_pd_f = np.argmax(fc8_value_f)
            if int(final_pd_1) == int(test_label_1):
                mean_acc_num += 1
            if int(final_pd_f) == int(test_label_1):
                final_acc_num += 1

    print("mean frame acc is %f" % np.mean(acc_values))
    print("%d->1 mean acc is %f" % (num_segments, float(mean_acc_num) / float(total_video_num)))
    print("final frame acc is %f" % (float(final_acc_num) / float(total_video_num)))

signal_act = False
def signal_handler(signal, frame):
    global signal_act
    signal_act = True

def choose_from(lst):
    item = None
    while item not in lst.keys():
        item = raw_input('Would you like to do? {}: '.format(lst.keys()))
    return lst[item]

signal.signal(signal.SIGINT, signal_handler)

if run_training:
    data_reader = ucf101.reader(root_dir, train_list, "RGB", batch_size, num_length, num_segments, False, "SEQ", queue_num=5)
    test_data_reader = ucf101.reader(root_dir, test_list, "RGB", batch_size, num_length, num_segments, True, "SEQ", queue_num=5)

    for i in range(start_step, total_steps):
        print("loading data")
        data, label = data_reader.get()
        print("running")
        (_, loss_value, acc_value, step, lr_value, summary) = sess.run([train_op, cross_entropy_loss, accuracy, global_step, lr, merged_summaries], feed_dict={batch_data:data, batch_label:label, lstm_keep_prob: lstm_keep_prob_value})
        if do_summary:
            summary_writer.add_summary(summary, step)
        print("[step %d]: loss, %f; acc, %f; lr, %f; lstm_keep, %f" % (step, loss_value, acc_value, lr_value, lstm_keep_prob_value))
        if step % test_inteval == 0: # or step == 1:
            print("testing ... ")
            all_acc = []
            all_loss = []
            all_fc8 = []
            all_labels = []
            for k in range(test_iter):
                test_data, test_label = test_data_reader.get()
                (loss_value, acc_value, fc8_value) = sess.run([cross_entropy_loss, accuracy, softmax_digits], feed_dict={batch_data:test_data, batch_label:test_label, lstm_keep_prob: 1.0})
                all_acc.append(acc_value)
                all_loss.append(loss_value)
                all_fc8.append(fc8_value)
                all_labels.append(test_label)
                print("testing iter %d" % (k))
            analysis_test_result(num_segments, test_iter, all_loss, all_acc, all_fc8, all_labels)

        if signal_act:
            action = choose_from({"save": "save", "summary": "summary", "exit":"exit"})
            print("choosing {}".format(action))
            if action == "save":
                print("Saving model")
                save_path = saver.save(sess, "./weights_rgb_lstm/rgb_vgg16_iter%d.ckpt" % (step))
                print("Model saved in file: %s" % save_path)
            elif action == "summary":
                do_summary = choose_from({"yes": True, "no": False})
                print("done, do_summary: {}".format(do_summary))
            print("exit ?")
            do_exit = choose_from({"yes": True, "no": False})
            if do_exit:
                sys.exit(0)
            else:
                signal_act = False

        if step % save_inteval == 0: #or step == 1:
            print("Saving model")
            save_path = saver.save(sess, "./weights_rgb_lstm/rgb_vgg16_iter%d.ckpt" % (step))
            print("Model saved in file: %s" % save_path)
        if step >= total_steps:
            break
    print("trainning finished")

if run_full_test:
    full_test_data_reader = ucf101.reader(root_dir, test_list, "RGB", batch_size, num_length, num_segments, True, "SEQ")
    full_test_video_num = full_test_data_reader.get_video_num()
    print("full testing ... ... ")
    print("totally test %d videos" % full_test_video_num)
    all_acc = []
    all_loss = []
    all_fc8 = []
    all_labels = []
    for i in range(full_test_video_num // batch_size):
        step = i + 1
        test_data, test_label = full_test_data_reader.get()
        (loss_value, acc_value, fc8_value) = sess.run([cross_entropy_loss, accuracy, softmax_digits], feed_dict={batch_data:test_data, batch_label:test_label, lstm_keep_prob: 1.0})
        tested_video_num = step * batch_size
        all_acc.append(acc_value)
        all_loss.append(loss_value)
        all_fc8.append(fc8_value)
        all_labels.append(test_label)
        if step % 10 == 0:
            analysis_test_result(num_segments, step, all_loss, all_acc, all_fc8, all_labels)
