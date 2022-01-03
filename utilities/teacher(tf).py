import tensorflow as tf
import numpy as np
import os
import vgg19
import time
from DatasetController import DatasetController
import os
from PIL import Image

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'



batchsize = 64
maxStringLen = 10
nb_frames = 1
h = 224
w = 224
vocabularySize = 50
embeddingSize = 100
c = 3
experiment_name = "ugp_teacher_module"
graph = tf.Graph()

with graph.as_default():	# everything inside this will run at each iteration, tf uses static graphs, will compute everytime
    with tf.variable_scope('main', initializer = tf.contrib.layers.xavier_initializer()):
        input_txt = tf.placeholder(dtype=tf.int32, shape=[batchsize, maxStringLen], name='InputSentence')
        input_video = tf.placeholder(dtype=tf.float32, shape=[batchsize, nb_frames, h, w, c], name='Input_Video')
        GT = tf.placeholder(dtype=tf.float32, shape=[batchsize,vocabularySize], name='groundTruth')
        dropoutRate = tf.placeholder(dtype=tf.float32, shape=[], name='dropoutRate')
        input_is_training = tf.placeholder(dtype = tf.bool, shape = [], name = 'isTraining')
        ## sentence embedding
        word_embeddings = tf.get_variable('WordEmbeddings', [vocabularySize, embeddingSize], tf.float32)

        embedded_words_sentence = tf.nn.embedding_lookup(word_embeddings, input_txt, max_norm=1)

        with tf.variable_scope('LSTM'):
            Cell = tf.nn.rnn_cell.LSTMCell(300)
            Encoded, _ = tf.nn.dynamic_rnn(Cell, embedded_words_sentence, dtype=tf.float32)
        Encoded = Encoded[:, -1, :]
        # Encoded = tf.nn.dropout(Encoded, dropoutRate)
        ## Video Encoddings

        # flattened_video = tf.layers.flatten(input_video)
        # flattened_video = tf.layers.batch_normalization(flattened_video, axis = -1, scale=False)
        # flattened_video = tf.reshape(flattened_video,[batchsize, nb_frames, h, w, c])
        


        vgg = vgg19.Vgg19()
        allFrames = tf.reshape(input_video,[batchsize*nb_frames,h,w,c])
        vgg.build(allFrames)
        video_features = vgg.conv4_4
        video = tf.reshape(video_features,[batchsize,nb_frames,video_features.get_shape()[1],video_features.get_shape()[2],video_features.get_shape()[3]])
        
        flattened_video = tf.layers.flatten(video)
        # print('remove following if didnt work')
        # flattened_video = tf.layers.batch_normalization(flattened_video, training = False, axis = -1, scale=False)
        flattened_video = tf.reshape(video_features,[batchsize,nb_frames,video_features.get_shape()[1],video_features.get_shape()[2],video_features.get_shape()[3]])
        video = flattened_video


        # video = tf.contrib.layers.batch_norm(input_video, updates_collections=None, decay=0.9)
        # video = tf.layers.conv3d(video, 256, (2, 3, 3), padding='same')
        # video = tf.contrib.layers.batch_norm(video, updates_collections=None, decay=0.9)
        # video = tf.nn.relu(video)
        #
        # video = tf.nn.dropout(video, dropoutRate)
        #
        # video = tf.layers.conv3d(video, 256, (2, 4, 4), padding='same')
        # # video = tf.contrib.layers.batch_norm(video, updates_collections=None, decay=0.9)
        # video = tf.layers.max_pooling3d(video, (1, 8, 8), (1, 2, 2), padding='same')
        # video = tf.layers.conv3d(video, 256, (2, 5, 5), padding='same')
        #
        # video = tf.contrib.layers.batch_norm(video, updates_collections=None, decay=0.9)
        # video = tf.nn.relu(video)
        #
        # video = tf.layers.conv3d(video, 512, (2, 8, 8), padding='same')
        # video = tf.layers.max_pooling3d(video, (1, 2, 2), (1, 2, 2), padding='same')
        #
        video = tf.reduce_max(video, 1)
        video = tf.layers.batch_normalization(video,training = input_is_training, axis= -1, scale=False)

        video = tf.reshape(video, [batchsize, video_features.get_shape()[1] * video_features.get_shape()[2], video_features.get_shape()[3]])

        video = tf.layers.dense(video, 512, use_bias = False)
        video = tf.layers.batch_normalization(video,training = input_is_training, axis= -1, scale=False)
        video = tf.nn.tanh(video)


        video_original = video
        # Combine Videos and Text

        Encoded = tf.layers.dense(Encoded, 32, use_bias = False)
        Encoded = tf.layers.batch_normalization(Encoded, training=input_is_training, axis=-1)
        Encoded = tf.nn.tanh(Encoded)


        video = tf.layers.dense(video, 32, use_bias = False)
        video = tf.layers.batch_normalization(video, training=input_is_training, axis=-1, scale=False)
        video = tf.nn.tanh(video)


        Encoded = tf.layers.dense(Encoded, 100)
        video = tf.layers.dense(video, 100)
        Encoded = tf.expand_dims(Encoded, 1)
        Encoded = tf.tile(Encoded, [1, video.get_shape()[1], 1])

        # Encoded = tf.nn.l2_normalize(Encoded, -1)
        # video = tf.nn.l2_normalize(video, -1)
        # Att = tf.einsum('ijk,ikm->ijm',video, Encoded)


        combined = video + Encoded
        combined = tf.layers.batch_normalization(combined, training=input_is_training, axis=-1, scale=False)
        combined = tf.nn.tanh(combined)
        Att = tf.layers.dense(combined, 1)     # takes the "combined" tensor, creates a dense nn(with just the one layer), with output dimensionality = 1
        Att = tf.squeeze(Att, -1)
        Att = tf.nn.softmax(Att, name='ComputeAttention', dim=-1)   # applying softmax, as used in the paper.

        features = tf.einsum('ijk,ij->ik', video_original, Att)
        features = tf.nn.dropout(features, dropoutRate)
        
        # features = tf.expand_dims(features,axis = 1)
        # features = tf.tile(features, [1,maxStringLen,1])
        features = tf.layers.dense(features,100, use_bias = False)
        features = tf.layers.batch_normalization(features,axis = -1)
        features = tf.layers.dense(features, vocabularySize)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=GT, logits=features)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(loss)
            #  train_step = tf.train.RMSPropOptimizer(learning_rate=0.001,momentum=0.9).minimize(loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
resume = 1
def show_image(images):
    # for i in range(images.shape[0]):
        # moving axis to use plt: i.e [4,100,100] to [100,100,4]
    img = images
    # img = img.transpose(1, 2, 0)
    img = (img +1) * 127.5
    img = img.astype(np.uint8)
    print img.dtype, np.max(img), np.min(img), np.shape(img)
    img = Image.fromarray(img, "RGB")
    img.show()

with tf.Session(graph=graph, config=config) as sess:
    if resume:
        try:
            new_saver = tf.train.Saver()
            new_saver.restore(sess, './models/'+experiment_name+'/'+experiment_name)
            print('saved model has been loaded***********************')
        except:
            sess.run(tf.global_variables_initializer())
            pass
    else:
        sess.run(tf.global_variables_initializer())
    print('initialized')
    # input_txt = tf.placeholder(dtype=tf.int32, shape=[batchsize, maxStringLen], name='InputSentence')
    # input_video = tf.placeholder(dtype=tf.float32, shape=[batchsize, nb_frames, h, w, c], name='Input_Video')
    # GT = tf.placeholder(dtype=tf.float32, shape=[batchsize], name='groundTruth')
    # dropoutRate = tf.placeholder(dtype=tf.float32, shape=[], name='dropoutRate')

    DC = DatasetController(batch_size=batchsize, sequence_input=nb_frames, sequence_output=0, string_size=maxStringLen, read_jpgs=True)
    g = DC.get_next_batch(task=['5001', '5002'], sth_sth=False, channel_first=False, human=False, camera='camera-1', train=True, attention_correct_prob=1.0)

    print('training starts')
    for iter in range(1000000):
        images, _, _, _, _, _, _, _, _, labels, gt, objects = next(g)
        # images = np.concatenate((robot_images, human_images), axis=0)

        gt_vocab = np.zeros((batchsize,vocabularySize))
        for b in range(labels.shape[0]):
            for w in labels[b,:]:
                gt_vocab[b,int(w)] = 1
        # print(gt_vocab.shape)
        # labels = np.repeat(labels, 2, axis=0)
        # gt_vocab = np.repeat(gt_vocab, 2, axis=0)

        # the following sess.run  runs 3 subgraphs, first is to execute the optimizer , second to calculate loss, and third to calculate the attention map 
        _, out_loss, attention = sess.run([train_step, loss, Att], feed_dict={input_txt: labels,
                                                         input_video: images,
                                                         GT: gt_vocab,
                                                         dropoutRate: 0.5, input_is_training:True})


        if (iter > 1 and iter % 100 == 0):
            images, paths, _, _, _, _, _, _, _, labels, gt, objects = next(g)
            # images = np.concatenate((robot_images, human_images), axis=0)
            # paths = np.concatenate((robot_paths, human_paths), axis=0)

            gt_vocab = np.zeros((batchsize,vocabularySize))
            for b in range(labels.shape[0]):
                for w in labels[b,:]:
                    gt_vocab[b,int(w)] = 1

            # labels = np.repeat(labels, 2, axis=0)
            # gt_vocab = np.repeat(gt_vocab, 2, axis=0)
            out_loss, attention = sess.run([loss, Att], feed_dict={input_txt: labels,
                                                                              input_video: images,
                                                                              GT: gt_vocab,
                                                                              dropoutRate: 1, input_is_training:False})
            print('saving$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

            saver = tf.train.Saver()
            # print(J[-99:])
            # print(np.mean(history[-99:]))
            try:
                os.stat('./models/' + experiment_name + '/')
            except:
                os.mkdir('./models/' + experiment_name + '/')
            saver.save(sess, './models/' + experiment_name + '/' + experiment_name)
            print (str(np.mean(out_loss)) + ' in iteration '+str(iter))
            print('saved#############################')

            # batch_input, reconstructed = sess.run([input_batch, FinalReconstructedFrames])
            np.save('./Results/paths_cap_cam_1', paths)
            np.save('./Results/attentions_cap_cam_1', attention)
            np.save('./Results/gt_cap_cam_1', gt)
