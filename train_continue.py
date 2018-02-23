import os
import tensorflow as tf
import inference
import train
import numpy as np

BATCH_SIZE = 256
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.9999
REGULARIZATION_RATE = 0.000001
TRAINING_STEPS = 192*1000000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"
SUMMARY_DIR = "./log/"

def train_continue(train_data, train_labels):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(
            tf.float32, [None, inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(
            tf.float32, [None, inference.OUTPUT_NODE], name='y-input')
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        y = inference.inference(x, regularizer)
        global_steps = tf.Variable(0, trainable=False)
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_steps)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())
        error = tf.reduce_mean(tf.square(y - y_)) + tf.add_n(tf.get_collection('errors'))
        tf.summary.scalar('mse', error)
        learning_rate = 0.0000001
        # learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_steps, 192, LEARNING_RATE_DECAY, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(error, global_step=global_steps)

        summary_op = tf.summary.merge_all()
        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')
        saver = tf.train.Saver()
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
            ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
#                global_steps = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            for i in range(TRAINING_STEPS):
                start = (i*BATCH_SIZE) % len(train_data)
                end = min(start+BATCH_SIZE, len(train_data))
                xs = train_data[start:end]
                ys = train_labels[start:end]
                [_, summary, error_value, step] = sess.run([train_op, summary_op, error, global_steps],
                                                           feed_dict={x: xs, y_: ys})
                if i % 19200 == 0:
                    [y_value] = sess.run([y], feed_dict={x: train_data})
                    print("After %d training steps, error is %g, learning rate is %g" % (
                    step, error_value, learning_rate))
                    print(y_value)
                    np.savetxt("./self_predict/predict_stress.txt", y_value)
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_steps)
                    summary_writer.add_summary(summary, i)
            summary_writer.close()
                
def main(argv=None):
    inputs = []
    with open('new_input_layer.dat') as f1:
        for line in f1.readlines():
            temp = line.split()
            temp2 = []
            for temp1 in temp:
                temp2.append(float(temp1))
            inputs.append(temp2)
    labels = []
    with open('new_output_layer.dat') as f2:
        for line in f2.readlines():
            temp = line.split()
            temp2 = []
            for temp1 in temp:
                temp2.append(float(temp1))
            labels.append(temp2)

    train_data = inputs
    train_labels = labels
    # train_labels = np.array(train_labels)
    # train_labels = train_labels[:, 0]
    # train_labels = np.reshape(train_labels, (-1, 1))
    train_continue(train_data, train_labels)

if __name__ == '__main__':
    tf.app.run()                
