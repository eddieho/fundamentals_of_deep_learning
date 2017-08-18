# Fundamentals of Deep Learning - TensorFlow v1.2.x porting

This is a port of sample code at https://github.com/darksigma/Fundamentals-of-Deep-Learning-Book to TensorFlow v1.2.x. The author of "Fundamentals of Deep Learning" wrote TensorFlow code on TF 1.0 or 1.1 and hopefully some trivial renaming of API calls will work. I have already ported CNN related scripts (conv*.py and cifar10_input.py) when I created this repo. I want to keep track of the latest port here and happy to share with any one reading the same book. 

Networks

    Logistic Regression (Nikhil)
    Multilayer Perceptron (Nikhil)
    Convolutional Network (Nikhil) - done
    Neural Style (Anish)
    Autoencoder (Hassan)
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
|Original|TF v1.2.x compatible|
|---|---|
|tf.image_summary("filters", V_T, max_images=64)|tf.summary.image("filters", V_T, max_outputs=64)|
|tf.nn.sparse_softmax_cross_entropy_with_logits(output, tf.cast(y, tf.int64))|tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=tf.cast(y, tf.int64))|
|tf.scalar_summary("cost", cost)|tf.summary.scalar("cost", cost)|
|tf.merge_all_summaries()|tf.summary.merge_all()|
|tf.train.SummaryWriter("conv_cifar_logs/",graph_def=sess.graph_def)|tf.summary.FileWriter("tf_events/conv_cifar_logs/", tf.get_default_graph())|
|tf.initialize_all_variables()|tf.global_variables_initializer()|

