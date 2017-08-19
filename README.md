# Fundamentals of Deep Learning - TensorFlow v1.2.x/Python 3.x porting

This is a port of sample code at https://github.com/darksigma/Fundamentals-of-Deep-Learning-Book to TensorFlow v1.2.x/Python 3.6.x. The author of "Fundamentals of Deep Learning" wrote TensorFlow code on TF 1.x (or pre-1.0?)/Python 2.7. Hopefully some trivial renaming of API calls will work. I have already ported CNN related scripts (conv*.py and cifar10_input.py) when I created this repo. I want to keep track of the latest port here and happy to share with any one reading the same book. 

Networks

    Logistic Regression (Nikhil)
    Multilayer Perceptron (Nikhil)
    Convolutional Network (Nikhil) - done
    Neural Style (Anish)
    Autoencoder (Hassan) - done
    Denoising Autoencoder (Hassan)
    Convolutional Autoencoder (Hassan) 
    RNN (Nikhil)
    LSTM Network (Nikhil)
    GRU Network (Nikhil)
    LSTM + Attention (Nikhil)
    RCNN (Nikhil)
    Memory Networks (Nikhil)
    Pointer Networks
    Neural Turing Machines
    Neural Programmer
    DQN
    LSTM-DQN
    Deep Convolutional Inverse Graphics Network
    Highway Networks
    Deep Residual Networks

Embedding

    Word2Vec (Nikhil)
    Skip-gram/CBoW
    GloVe (Nikhil)
    Skip-thought Vectors (Nikhil)

Optimizers

    MLP + Momentum
    MLP + RMSProp
    MLP + ADAM
    MLP + FTRL
    MLP + ADADELTA


## Summary of changes (examples)
|Original/Python 2.7|TF v1.2.x/Python 3.x compatible|
|---|---|
|tf.image_summary("filters", V_T, max_images=64)|tf.summary.image("filters", V_T, max_outputs=64)|
|tf.nn.sparse_softmax_cross_entropy_with_logits(output, tf.cast(y, tf.int64))|tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=tf.cast(y, tf.int64))|
|tf.scalar_summary("cost", cost)|tf.summary.scalar("cost", cost)|
|tf.merge_all_summaries()|tf.summary.merge_all()|
|tf.train.SummaryWriter("conv_cifar_logs/",graph_def=sess.graph_def)|tf.summary.FileWriter("tf_events/conv_cifar_logs/", tf.get_default_graph())|
|tf.initialize_all_variables()|tf.global_variables_initializer()|
|tf.sub()|tf.subtract()|
|tf.mul()|tf.multiply()|
|from tensorflow.python import control_flow_ops|from tensorflow.python.ops import control_flow_ops|
|tf.nn.nce_loss(weights, biases, inputs, labels,...)|tf.nn.nce_loss(weights, biases, labels, inputs,...) - not sure if this is an erratta even on pre-1.0 TF. Use named arguments in ported skipgram.py|
|Python 2.x print|Python 3.x style print("string".format(val1,val2,...)|
|xrange(M,N) - Python 2.7|range(M,N) - Python 3.x|

In addition to mandatory changes due to TensorFlow v1.2.x, the following changes have been made:
1. Use config to control TF session's memory usage
By default, TensorFlow grabs all available memory from GPU device and a TF session would use the GPU exclusively. If these lines are used, then the TF session will only use memory as needed. Let's say a GPU has 8GB memory. Without this parameter, a TF session will get about 7.xGB. If your TF code only needs 300MB memory with this parameter, then TF session will only grab 300MB GPU memory. You can verify the difference by the command nvidia-smi on Nvidia GPU.

runtimeConfig = tf.ConfigProto()

runtimeConfig.gpu_options.allow_growth = True

sess = tf.Session(config=runtimeConfig)


