import tensorflow as tf
import numpy as np
from word_id_test import Word_Id_Map


with tf.device('/cpu:0'):
    batch_size = 1
    sequence_length = 20
    num_encoder_symbols = 1004
    num_decoder_symbols = 1004
    hidden_size = 256
    embedding_size = 256
    num_layers = 2

    # 创建placeholder
    encoder_inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])
    decoder_inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])

    targets = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])
    weight = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])

    # 创建cell单元
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

    # 创建RNN链
    # results, last_stat = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
    #     tf.unstack(encoder_inputs, axis=1),
    #     tf.unstack(decoder_inputs, axis=1),
    #     cell,
    #     num_encoder_symbols,
    #     num_decoder_symbols,
    #     embedding_size,
    #     feed_previous=False
    # )
    results, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
        tf.unstack(encoder_inputs, axis=1),
        tf.unstack(decoder_inputs, axis=1),
        cell,
        num_encoder_symbols,
        num_decoder_symbols,
        embedding_size,
        # decoder端是否将前一时刻的预测作为输入（在Train时，不会将前一时刻作为输入）
        feed_previous=True,
    )

    logtis = tf.stack(results, axis=1)
    pred = tf.argmax(logtis, axis=2)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # model_file = tf.train.latest_checkpoint('./model/')
        module_file = tf.train.latest_checkpoint('./model/')
        saver.restore(sess, module_file)
        map = Word_Id_Map()
        encoder_input = map.sentence2ids(['how', 'do', 'you', 'do', 'this'])
        # encoder_input = encoder_input + [3 for i in range(0, 20 - len(encoder_input))]
        encoder_input = encoder_input + [3 for i in range(0, 20 - len(encoder_input))]
        encoder_input = np.asarray([np.array(encoder_input)])
        decoder_input = np.zeros([1, 20])

        pred_value = sess.run(pred, feed_dict={encoder_inputs: encoder_input, decoder_inputs: decoder_input})
        sentence = map.ids2sentence(pred_value[0])
        print(" ".join(sentence))






















