import tensorflow as tf
import config

class Generator():
    def __init__(self, n_node, n_relation, node_emd_init, relation_emd_init):
        self.n_node = n_node
        self.n_relation = n_relation
        self.node_emd_init = node_emd_init
        self.relation_emd_init = relation_emd_init
        self.emd_dim = node_emd_init.shape[1]

        #with tf.variable_scope('generator'):
        self.node_embedding_matrix = tf.get_variable(name = "gen_node_embedding",
                                                     shape = self.node_emd_init.shape,
                                                     initializer = tf.constant_initializer(self.node_emd_init),
                                                     trainable = True)
        self.relation_embedding_matrix = tf.get_variable(name = "gen_relation_embedding",
                                                         shape = [self.n_relation, self.emd_dim, self.emd_dim],
                                                         initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                                         trainable = True)

        self.gen_w_1 = tf.get_variable(name = 'gen_w',
                                       shape = [self.emd_dim, self.emd_dim],
                                       initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                       trainable = True)
        self.gen_b_1 = tf.get_variable(name = 'gen_b',
                                       shape = [self.emd_dim],
                                       initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                       trainable = True)
        self.gen_w_2 = tf.get_variable(name = 'gen_w_2',
                                       shape = [self.emd_dim, self.emd_dim],
                                       initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                       trainable = True)
        self.gen_b_2 = tf.get_variable(name = 'gen_b_2',
                                       shape = [self.emd_dim],
                                       initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                       trainable = True)
        #self.bias_vector = tf.Variable(tf.zeros([self.n_node]))

        self.node_id =  tf.placeholder(tf.int32, shape = [None])
        self.relation_id = tf.placeholder(tf.int32, shape = [None])
        self.noise_embedding = tf.placeholder(tf.float32, shape = [None, self.emd_dim])

        self.dis_node_embedding = tf.placeholder(tf.float32, shape = [None, self.emd_dim])
        self.dis_relation_embedding = tf.placeholder(tf.float32, shape = [None, self.emd_dim, self.emd_dim])

        self.node_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, self.node_id)
        self.relation_embedding = tf.nn.embedding_lookup(self.relation_embedding_matrix, self.relation_id)
        self.node_neighbor_embedding = self.generate_node(self.node_embedding, self.relation_embedding, self.noise_embedding)

        t = tf.reshape(tf.matmul(tf.expand_dims(self.dis_node_embedding, 1), self.dis_relation_embedding), [-1, self.emd_dim])
        self.score = tf.reduce_sum(tf.multiply(t, self.node_neighbor_embedding), axis = 1)

        self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.score) * (1.0 - config.label_smooth), logits=self.score)) \
                  + config.lambda_gen * (tf.nn.l2_loss(self.node_embedding) + tf.nn.l2_loss(self.relation_embedding) + tf.nn.l2_loss(self.gen_w_1))

        optimizer = tf.train.AdamOptimizer(config.lr_gen)
        #optimizer = tf.train.RMSPropOptimizer(config.lr_gen)
        self.g_updates = optimizer.minimize(self.loss)

    def generate_node(self, node_embedding, relation_embedding, noise_embedding):
        #node_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, node_id)
        #relation_embedding = tf.nn.embedding_lookup(self.relation_embedding_matrix, relation_id)

        input = tf.reshape(tf.matmul(tf.expand_dims(node_embedding, 1), relation_embedding), [-1, self.emd_dim])
        #input = tf.concat([input, noise_embedding], axis = 1)
        input = input + noise_embedding

        output = tf.nn.leaky_relu(tf.matmul(input, self.gen_w_1) + self.gen_b_1)
        #input = tf.nn.leaky_relu(tf.matmul(input, self.gen_w_1) + self.gen_b_1)# +  relation_embedding
        #output = tf.nn.leaky_relu(tf.matmul(input, self.gen_w_2) + self.gen_b_2)
        #output = node_embedding + relation_embedding + noise_embedding

        return output
