import numpy as np
import tensorflow as tf
from utils_OAI import CategoricalPd, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
# from baselines.common.distributions import make_pdtype

# def nature_cnn(unscaled_images):
#     """
#     CNN from Nature paper.
#     """
#     scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
#     activ = tf.nn.relu
#     h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
#     h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
#     h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
#     h3 = conv_to_fc(h3)
#     return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

def fc_layers(game_state):  # TODO decide on architecture
    activ = tf.tanh
    h = activ(fc(game_state, 'fc1', nh=64)) #, init_scale=np.sqrt(2)))
    h1 = activ(fc(h, 'fc2', nh=64)) # , init_scale=np.sqrt(2)))
    return activ(fc(h1, 'fc3', nh=32) )#, init_scale=np.sqrt(2)))

def random_choice(sess, data, probs):
    data = tf.convert_to_tensor(data)
    assert data.shape == probs.shape, 'array and probability need to have the same shape'
    idx_sample = tf.multinomial(tf.log(probs), 1)
    return data[tf.cast(idx_sample[0][0], tf.int32)].eval(session=sess)

class LnLstmPolicy(object):
    def __init__(self, sess, ob_shape, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        # this method is called with nbatch = nenvs*nsteps
        nenv = nbatch // nsteps  # nenvs entspricht der Anzahl an nstep-trajectorien, die aus dem replay buffer verwendet werden = replay_batch

        # Input and Output dimensions
        nd, = ob_shape  # nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nd)  # ob_shape = (nbatch, nh, nw, nc)
        nact = len(ac_space)  # nact = ac_space.n

        X = tf.placeholder(tf.float32, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = fc_layers(X)  # nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)

        # self.pdtype = make_pdtype(ac_space)
        # self.pd = self.pdtype.pdfromflat(pi)
        self.pd = CategoricalPd(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class LstmPolicy(object):

    def __init__(self, sess, ob_shape, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        # this method is called with nbatch = nenvs*nsteps
        nenv = nbatch // nsteps  # nenvs entspricht der Anzahl an nstep-trajectorien, die aus dem replay buffer verwendet werden = replay_batch

        # nh, nw, nc = ob_space.shape
        # ob_shape = (nbatch, nh, nw, nc)
        # nact = len(ac_space)  # ac_space.n

        # Input and Output dimensions
        nd, = ob_shape
        ob_shape = (nbatch, nd)
        nact = len(ac_space)  # ac_space.n
        X = tf.placeholder(tf.float32, ob_shape, name='X') #obs
        M = tf.placeholder(tf.float32, [nbatch], name='M') #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2], name='S') #states
        with tf.variable_scope("model", reuse=reuse):
            h = fc_layers(X)  # nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)

        # self.pdtype = make_pdtype(ac_space)
        # self.pd = self.pdtype.pdfromflat(pi)
        self.pd = CategoricalPd(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class FCPolicy(object):

    def __init__(self, sess, ob_shape, ac_space, nbatch, nsteps, reuse=False):  # pylint: disable=W0613
        # this method is called with nbatch = nenvs*nsteps

        # nh, nw, nc = ob_space.shape  # no image but numerical values
        # ob_shape = (nbatch, nh, nw, nc)

        # Input and Output dimensions
        nd, = ob_shape
        ob_shape = (nbatch, nd)
        nact = len(ac_space)  # ac_space.n
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        with tf.variable_scope("model", reuse=reuse):
            h = fc_layers(X)  # nature_cnn(X)
            pi = fc(h, 'pi', nact, init_scale=0.01)
            vf = fc(h, 'v', 1)[:,0]

        # self.pdtype = make_pdtype(ac_space)
        # self.pd = self.pdtype.pdfromflat(pi)
        self.pd = CategoricalPd(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        # this method is called with nbatch = nenvs*nsteps

        # nh, nw, nc = ob_space.shape
        # ob_shape = (nbatch, nh, nw, nc)
        # actdim = ac_space.shape[0]

        # Input and Output dimensions
        nd, = ob_space.shape
        ob_shape = (nbatch, nd)
        nact = ac_space.n
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            h1 = activ(fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'pi_fc2', nh=32, init_scale=np.sqrt(2)))
            pi_logit = fc(h2, 'pi', nact, init_scale=0.01)
            pi = tf.nn.softmax(pi_logit)

            # h1 = activ(fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))  # TODO add these layers
            # h2 = activ(fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(h2, 'vf', 1)[:,0]
            logstd = tf.get_variable(name="logstd", shape=[1, nact],
                initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)
        # self.pdtype = make_pdtype(ac_space)
        # self.pd = self.pdtype.pdfromflat(pdparam)
        self.pd = CategoricalPd(pi)  # pdparam
        a0 = self.pd.sample()  # returns action index: 0,1
        # a0 = np.random.choice(ac_space, p=pi)  # returns action value: 119,None

        neglogp0 = self.pd.neglogp(a0)

        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, vf], {X:ob})
            return a, v, self.initial_state

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class CastaPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):  # pylint: disable=W0613
        # this method is called with nbatch = nenvs*nsteps

        # nh, nw, nc = ob_space.shape
        # ob_shape = (nbatch, nh, nw, nc)
        # actdim = ac_space.shape[0]
        # Todo check initialization
        # Input and Output dimensions
        nd, = ob_space.shape
        ob_shape = (nbatch, nd)
        nact = ac_space.n
        X = tf.placeholder(tf.float32, ob_shape, name='Ob')  # obs
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.elu
            h1 = activ(fc(X, 'pi_vf_fc1', nh=64, init_scale=np.sqrt(2)))

            h2 = activ(fc(h1, 'pi_fc2', nh=32, init_scale=np.sqrt(2)))
            pi_logit = fc(h2, 'pi', nact, init_scale=0.01)
            pi = tf.nn.softmax(pi_logit)

            # h4 = activ(fc(h1, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))  # TODO add these layers
            h5 = activ(fc(h1, 'vf_fc2', nh=32, init_scale=np.sqrt(2)))
            vf = fc(h5, 'vf', 1)[:, 0]
            logstd = tf.get_variable(name="logstd", shape=[1, nact],
                                     initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)
        # self.pdtype = make_pdtype(ac_space)
        # self.pd = self.pdtype.pdfromflat(pdparam)
        self.pd = CategoricalPd(pi_logit)  # pdparam
        a0 = self.pd.sample()  # returns action index: 0,1
        # a0 = np.random.choice(ac_space, p=pi)  # returns action value: 119,None
        # a0 = random_choice(sess, np.ones(shape=(nbatch, nact)) * np.array(range(nact)).T, pi)
        # neglogp0 = self.pd.neglogp(a0)

        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, vf], {X: ob})
            return a, v, self.initial_state

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


# -------------------------------------------------------------------------------------------
#                                    ACER Policies
# -------------------------------------------------------------------------------------------

class AcerCnnPolicy(object):

    def __init__(self, sess, ob_shape, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv * nsteps
        # nh, nw, nc = ob_space.shape
        # ob_shape = (nbatch, nh, nw, nc * nstack)
        # nact = ac_space.n
        nd, = ob_shape
        ob_shape = (nbatch, nd)
        nact = len(ac_space)
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        with tf.variable_scope("model", reuse=reuse):
            h = fc_layers(X)  # nature_cnn(X)
            pi_logits = fc(h, 'pi', nact, init_scale=0.01)
            pi = tf.nn.softmax(pi_logits)
            q = fc(h, 'q', nact)

        # a = np.random.choice(ac_space, p=pi)  # sample(pi_logits)  # could change this to use self.pi instead
        a = random_choice(ac_space, pi)
        self.initial_state = []  # not stateful
        self.X = X
        self.pi = pi  # actual policy params now
        self.q = q

        def step(ob, *args, **kwargs):
            # returns actions, mus, states
            a0, pi0 = sess.run([a, pi], {X: ob})
            return a0, pi0, []  # dummy state

        def out(ob, *args, **kwargs):
            pi0, q0 = sess.run([pi, q], {X: ob})
            return pi0, q0

        def act(ob, *args, **kwargs):
            return sess.run(a, {X: ob})

        self.step = step
        self.out = out
        self.act = act

class AcerLstmPolicy(object):

    def __init__(self, sess, ob_shape, ac_space, nenv, nsteps, nstack, reuse=False, nlstm=256):
        nbatch = nenv * nsteps
        # nh, nw, nc = ob_space.shape
        # ob_shape = (nbatch, nh, nw, nc * nstack)
        # nact = ac_space.n
        nd, = ob_shape
        ob_shape = (nbatch, nd)
        nact = len(ac_space)  # ac_space.n
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = fc_layers(X)  # nature_cnn(X)

            # lstm
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)

            pi_logits = fc(h5, 'pi', nact, init_scale=0.01)
            pi = tf.nn.softmax(pi_logits)
            q = fc(h5, 'q', nact)

        # a = np.random.choice(ac_space, p=self.pi)  # sample(pi_logits)  # could change this to use self.pi instead
        a = random_choice(ac_space, pi)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)
        self.X = X
        self.M = M
        self.S = S
        self.pi = pi  # actual policy params now
        self.q = q

        def step(ob, state, mask, *args, **kwargs):
            # returns actions, mus, states
            a0, pi0, s = sess.run([a, pi, snew], {X: ob, S: state, M: mask})
            return a0, pi0, s

        self.step = step


# -------------------------------------------------------------------------------------------------
#                                           DQN
# -------------------------------------------------------------------------------------------------