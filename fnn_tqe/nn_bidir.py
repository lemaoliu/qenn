from collections import OrderedDict
import cPickle as pkl
import sys
import time
import logging
import re

sys.path.append('/panfs/panmt/users/lliu/code/qe-lstm')

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from imdb import preprocess_data, prepare_reorderdata_minibatch, \
                 get_label, load_dict, seq2indices
from imdb import split_train,construct_vcb
import imdb
import os
from theano.ifelse import ifelse

from search import BeamSearch,EnsembleBeamSearch
import theano.sandbox.cuda
#theano.sandbox.cuda.use('gpu3')

logger = logging.getLogger(__name__)
SEED = 123
# fix the random seed as constant=SEED
#numpy.random.seed(SEED)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)



def softmax(x):

    if x.ndim == 2:
        e = tensor.exp(x)
        return e / tensor.sum(e, axis=1).dimshuffle(0, 'x')
    elif x.ndim == 3:
        e = tensor.exp(x)
        return e / tensor.sum(e, axis=2).dimshuffle(0, 1,'x')
    elif x.ndim == 1:
        e = tensor.exp(x)
        return e/ tensor.sum(e)
    else:
        die


def numpy_int(data):
    return numpy.asarray(data, dtype='int64')


def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n * dim:(n + 1) * dim]
    else:
        return _x[:, n * dim:(n + 1) * dim]


def _p(prefix,pre):
        return "%s_%s"%(prefix,pre)


def print_tparams(tparams):
    print 'the sum'
    for v in tparams:
        print v.name,v.get_value().sum(),
    print ''



class Trainer(object):

    def __init__(self, coeff=0.95, diag=1e-6, eta=2):
        self.coeff = coeff
        self.diag = diag
        self.eta = eta
        self.use_sgd = 0

    def sgd(self, lr, inputs, cost, grads, tparams):
        """ Stochastic Gradient Descent

        :note: A more complicated version of sgd then needed.  This is
            done like that for adadelta and rmsprop.

        """
        # New set of shared variable that will contain the gradient
        # for a mini-batch.
        gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % p.name)
                   for p in tparams]
        gsup = [(gs, g) for gs, g in zip(gshared, grads)]

        # Function that computes gradients for a mini-batch, but do not
        # updates the weights.
        f_grad_shared = theano.function(inputs, cost, updates=gsup,
                                        name='sgd_f_grad_shared')

        pup = [(p, p - lr * g) for p, g in zip(tparams, gshared)]

        # Function that updates the weights from the previously computed
        # gradient.
        f_update = theano.function([lr], [], updates=pup,
                                   name='sgd_f_update')

        return f_grad_shared, f_update


    def SetupTainer(self, lr, inputs, cost, tparams):

        grads = tensor.grad(cost, wrt=tparams, disconnected_inputs='ignore')
        #print "grads",type(grads), type(grads[0])
        #print "grads",map(lambda x: x.name, grads)
        #print "tparams",tparams
        grads = clip_grad(grads,tparams)

        if self.use_sgd:
            return self.sgd(lr, inputs, cost, grads, tparams)
        else:
            return self.adadelta(lr, inputs, cost, grads, tparams)

    def adadelta(self, lr, inputs, cost, grads, tparams):
        zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                      name='%s_grad' % p.name)
                        for p in tparams]
        running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                     name='%s_rup2' % p.name) ## delte_x_square_sum
                       for p in tparams]
        running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                        name='%s_rgrad2' % p.name)
                          for p in tparams] ### g_square_sum

        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]

        rg2up = [(rg2, self.coeff * rg2 + (1-self.coeff) * (g ** 2))
                 for rg2, g in zip(running_grads2, grads)]
        ### update the g_square_sum
        f_grad_shared = theano.function(inputs, cost, updates=zgup + rg2up,
                                        name='adadelta_f_grad_shared',on_unused_input='ignore')

        updir = [-tensor.sqrt(ru2 + self.diag) / tensor.sqrt(rg2 + self.diag) * zg
                 for zg, ru2, rg2 in zip(zipped_grads,
                                         running_up2,
                                         running_grads2)]
        ru2up = [(ru2, self.coeff * ru2 + (1-self.coeff) * (ud ** 2))
                 for ru2, ud in zip(running_up2, updir)]
        param_up = [(p, p + self.eta*ud) for p, ud in zip(tparams, updir)]

        ## update the delte_x_square_sum and parameters
        f_update = theano.function([lr], [], updates=ru2up + param_up,
                                   on_unused_input='ignore',
                                   name='adadelta_f_update')

        return f_grad_shared, f_update


def clip_grad(grads,params, max_norm=numpy_floatX(5.)):

    grad_norm = tensor.sqrt(sum(map(lambda x: tensor.sqr(x).sum(), grads)))
    not_finite = tensor.or_(tensor.isnan(grad_norm), tensor.isinf(grad_norm))
    grad_norm = tensor.sqrt(grad_norm)
    cgrads = []
    for g,p in zip(grads,params):
        tmpg = tensor.switch(tensor.ge(grad_norm, max_norm), g*max_norm/grad_norm, g)
        ttmpg = tensor.switch(not_finite, numpy_floatX(0.1)*p,tmpg)
        cgrads.append(ttmpg)

    return cgrads


class Layer(object):

    def __init__(self,dim,prefix=""):
        self.dim = dim
        self.prefix = self.prefix

    def pp(self,pre):
        return _p(self.prefix,pre)

    def ortho_weight(self,ndim):
        W = numpy.random.randn(ndim, ndim)
        u, s, v = numpy.linalg.svd(W)
        return u.astype(config.floatX)

    def ortho_weight_copies(self, d, n):
        Ws = [self.ortho_weight(d) for i in xrange(n)]
        return numpy.concatenate(Ws,axis=1)

    def rand_weight(self,shape):## shape is a tuple
        randn = numpy.random.rand(*shape)
        return (0.01 * randn).astype(config.floatX)


class Layer_emb(Layer):

    def __init__(self,dim,vcbsize,prefix=""):
        self.prefix = prefix
        self.dim = dim
        self.vcbsize = vcbsize
        self.emb_W = theano.shared(self.rand_weight((vcbsize,dim)),name=self.pp('emb'))
        self.params = [self.emb_W]

    ## x.shape=(timesteps,n)
    ## x is a list (sentence) or a list of list (minibatch of sentences)

    def reset_embd(self,embfile,vcb):
        ebd_w = self.emb_W.eval()
        for line in open(embfile):
            fields = line.strip().split()
            if len(fields) != self.dim + 1: 
                print line, 
                continue
            w, vec = fields[0], numpy.fromstring(' '.join(fields[1:]),sep=' ')
            assert vec.shape[0] == self.dim, 'error in embedding'
            if w in vcb:
                ebd_w[vcb[w]] = vec
        self.emb_W.set_value(numpy_floatX(ebd_w))
        numpy.savez('%s.num'%embfile,self.emb_W.eval())

    def fprop(self,x):
        #print "emb dim", self.dim
        emb = self.emb_W[x.flatten()]
        #print "x.ndim",x.ndim
        if x.ndim == 1:
            return emb
        else:
            return tensor.reshape(emb,(x.shape[0],x.shape[1],self.dim))
            #return emb.reshape((x.shape[0],x.shape[1],self.dim))

class Layer_linear(Layer):

    def __init__(self,in_dim,out_dim,active=None,prefix=""):
        self.prefix = prefix
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = theano.shared(self.rand_weight((in_dim,out_dim)),name=self.pp('w'))
        self.b = theano.shared(numpy.zeros((out_dim,)).astype(config.floatX),name=self.pp('b'))
        self.params = [self.W,self.b]
        self.active=active
    ### x is with shape=(t,n,in_dim)
    def fprop(self,x):
        z = tensor.dot(x,self.W) + self.b
        if self.active:
            z = tensor.nnet.sigmoid(z)
        return z 

class MLP(Layer):

    def __init__(self,dims,prefix=""):
        self.prefix = prefix
        self.dims = dims
        self.models = []
        self.params = []
        for i in xrange(1,len(dims)): 
            in_dim,out_dim = dims[i-1],dims[i]
            layer_i = Layer_linear(in_dim,out_dim,True,prefix+'_layer_%d'%(i-1))
            self.models.append(layer_i)
            self.params += layer_i.params

    def fprop(self,x):
        for m in self.models:
            x = m.fprop(x)
        return x

class feedforward(Layer):

    def __init__(self, _config, emb_x=None, emb_y=None, emb_label=None, prefix=''):
        ### basic config
        self.config = _config
        self.prefix = prefix
        self.dim = _config['dim_proj']
        self.srcvcbsize = _config['n_words_x']
        self.trgvcbsize = _config['n_words_y']
        self.labelvcbsize = _config['n_words_label']
        self.reverse_src = _config['reverse_src']
        self.reverse_trg = _config['reverse_trg'] if 'reverse_trg' in _config else False
        self.emb_x = emb_x if emb_x is not None else Layer_emb(self.dim,self.srcvcbsize,'src')
        self.emb_y = emb_y if emb_y is not None else Layer_emb(self.dim,self.trgvcbsize,'trg')
        if 'src_emb' in _config:
            self.emb_x.reset_embd(_config['src_emb'],_config['srcdict'])
        if 'trg_emb' in _config:
            self.emb_y.reset_embd(_config['trg_emb'],_config['trgdict'])
        self.emb_label = emb_label if emb_label is not None else Layer_emb(self.dim,self.labelvcbsize,'label')
        self.layers = _config['layers']
        self.ctx_size = _config['ctx_size']
        print 'context size',self.ctx_size
        self.mlp_dims = [2*self.ctx_size*self.dim] + [self.dim]*self.layers
        self.mlp =  MLP(self.mlp_dims,'mlp')

        self.U = theano.shared(self.rand_weight((self.mlp_dims[-1],self.emb_label.vcbsize)),name=_p(self.prefix,'U')) ### (d,self.emb.vcbsize)
        self.b = theano.shared(numpy.zeros((self.emb_label.vcbsize,)).astype(config.floatX), name=self.pp('b')) ### (1,self.emb.vcbsize)
        self.cost = None
        self.params  = []

        ### model params for training
        if emb_x is None:
            self.params = self.emb_x.params + self.emb_y.params + self.emb_label.params
        self.params += self.mlp.params + [self.U,self.b]

    def Build(self):
        use_noise = theano.shared(numpy_floatX(0.))
        x = tensor.matrix('x', dtype='int64') ## shape=(n,t,d), t is the ctx size
        y = tensor.matrix('y', dtype='int64')
        label = tensor.vector('label', dtype='int64')
        self.inputs = [x,y,label]
        logger.debug("To Build compuatation graph")
        enc_x = self.emb_x.fprop(x)
        enc_y = self.emb_y.fprop(y)
        self.x_repr_fn = theano.function([x],enc_x)
        enc_x = enc_x.reshape((enc_x.shape[0],enc_x.shape[1]*enc_x.shape[2]))
        enc_y = enc_y.reshape((enc_y.shape[0],enc_y.shape[1]*enc_y.shape[2]))
        ctx = tensor.concatenate((enc_x,enc_y),axis=1)## ctx.shape=(n,self.mlp_dims[0])
        proj = self.mlp.fprop(ctx)## proj.shape=(n,self.mlp_dims[-1])
        pred = softmax(tensor.dot(proj,self.U)+self.b)
        self.argpred = pred.argmax(axis=1)
        B = tensor.arange(y.shape[0])
        l_pred = pred[B,label]
        cost = -tensor.log(l_pred + 1e-8)
        self.cost = cost.sum()/cost.shape[0]
        logger.debug("Building compuatation graph over")
        self.f_cost = theano.function(self.inputs,self.cost,name='f_cost',on_unused_input='ignore')
        self.f_pred = theano.function([x,y],[pred[B,self.argpred],self.argpred],name='f_pred',on_unused_input='ignore')
        return self.inputs, self.cost, use_noise

    def show_norm_paras(self):
        for p in self.params:
            print 'norm of ', p, ' is ', numpy.linalg.norm(p.eval())

    def unzip(self):
        new_params = OrderedDict()
        for p in self.params:
            new_params[p.name] = p.get_value()
        return new_params

    def add_dict(self, ori_dict, adder_dict):
        for kk, vv in adder_dict.iteritems():
            ori_dict[kk] = vv

    def save(self,mfile,params=None, **info):
        new_params = self.unzip() if params is None else params
        self.add_dict(new_params,info)
        self.add_dict(new_params,self.config)
        numpy.savez(mfile,**new_params)

    def zipp(self,new_params):
        for p in self.params:
            p.set_value(numpy_floatX(new_params[p.name]))

    def load(self,mfile):
        new_params = numpy.load(mfile)
        self.zipp(new_params)
        '''
        for p in self.params:
            p.set_value(new_params[p.name])
        '''


class Lstm_layer(Layer):

    def __init__(self,dim,layers,prefix=""):
        #super(Layer, self).__init__(dim,prefix)
        self.dim = dim
        self.prefix = prefix
        self.layers = layers## it is not tensor variable
        self.W_hi = theano.shared(self.ortho_weight_copies(self.dim, 4*self.layers), name=self.pp('W_hi')) ### (d, layers*4*d), reuse the history hiden as input of lstm net, each _slice is used for one layer
        self.W_xi = theano.shared(self.ortho_weight_copies(self.dim, 4*self.layers),name=self.pp('W_xi')) ### (d, layers*4*d), reuse the input x as input of lstm net
        self.W_lhi = theano.shared(self.ortho_weight_copies(self.dim, 4*self.layers), name=self.pp('W_lhi')) ### (d, layers*4*d), reuse the lower hiden layer as input of lstm net
        self.bx = theano.shared(numpy.zeros((4*self.layers*self.dim,)).astype(config.floatX),name=self.pp('W_bx')) ### (1,layers*4*d)

        self.params = [self.W_hi,self.W_xi,self.W_lhi,self.bx]

    ### h_.shape=(n,layers*d), where n is the batch size
    ### x_.shape=(n,layers*4*d)
    ### c_.shape=(n,layers*d)
    def _step(self, m_, x_, h_, c_):
        n_samples = h_.shape[0]
        assert h_.ndim == 2, 'ndim error in _step func'
        hs, cs = [], []
        zeros = tensor.alloc(numpy_floatX(0.), n_samples, self.dim)

        for l in xrange(self.layers):
            ##  _slice(self.W_hi,l,4*self.dim), reuse the history as input, (d, 4*d)
            w_h = tensor.dot(_slice(h_,l,self.dim), _slice(self.W_hi,l,4*self.dim))
            ht_ = w_h + _slice(x_,l,4*self.dim)  ## shape=(n,4*d)
            hl_ = hs[-1] if len(hs) > 0 else zeros

            ## reuse the lower hiden layer as input of lstm net
            ht_ += tensor.dot(hl_,_slice(self.W_lhi,l,4*self.dim))
            ct_ = _slice(c_,l,self.dim)

            i = tensor.nnet.sigmoid(_slice(ht_, 0, self.dim))
            f = tensor.nnet.sigmoid(_slice(ht_, 1, self.dim))
            o = tensor.nnet.sigmoid(_slice(ht_, 2, self.dim))
            c = tensor.nnet.sigmoid(_slice(ht_, 3, self.dim))

            c = f * ct_ + i * c
            h = o * tensor.tanh(c)

            hs.append(h)
            cs.append(c)

        h = tensor.concatenate(hs,axis=1) ### (n,layers*d)
        c = tensor.concatenate(cs,axis=1) ### (n,layers*d)
        c = m_[:, None] * c + (1. - m_)[:, None] * c_
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    ### state_below is the lower representation,
    def fprop(self,state_below,mask=None,start_h=None,start_c=None):

        n_timesteps = state_below.shape[0]
        n_samples = state_below.shape[1]

        if start_h is None:
            start_h = tensor.alloc(numpy_floatX(0.), n_samples, self.layers*self.dim)
        if start_c is None:
            start_c = tensor.alloc(numpy_floatX(0.), n_samples, self.layers*self.dim)

        ## shape=(n,layers*4*d)
        state_below = tensor.dot(state_below,self.W_xi) + self.bx
        self.state_below = state_below
        rval, _ = theano.scan(self._step,
                                    sequences=[mask, state_below],
                                    outputs_info=[start_h,start_c],
                                    name=_p(self.prefix, '_layers'),
                                    n_steps=n_timesteps)
        return rval[0], rval[1] ### return h and c for all timesteps and all layers

    def BuildEnc(self,state_below,mask):
        all_h,all_c = self.fprop(state_below,mask)
        return all_h[-1], all_c[-1]

    '''
    def get_repr(self,state_below,mask=None):
        state_below = self.get_repr(state_below,mask)
        return state_below[-1]## shape=(n,d)
        return rval[0][:,:,beg:] #_slice(rval[0][-1],self.layers-1,self.dim) ### (t,n,d)
    '''



class Lstm_decoder(Lstm_layer):

    def __init__(self,dim,emb,layers,prefix="",use_linear_layer=False):
        self.sum_over_time = True
        #super(Lstm_layer, self).__init__(dim,layers,prefix)
        Lstm_layer.__init__(self,dim,layers,prefix)
        self.emb = emb
        self.U = theano.shared(self.rand_weight((self.dim,self.emb.vcbsize)),name=_p(self.prefix,'U')) ### (d,self.emb.vcbsize)
        self.b = theano.shared(numpy.zeros((self.emb.vcbsize,)).astype(config.floatX), name=self.pp('b')) ### (1,self.emb.vcbsize)
        self.params += [self.U,self.b]
        self.use_linear_layer = use_linear_layer
        if use_linear_layer:
            self.linear_layer = Layer_linear(2*self.dim,self.dim,'linear')
            self.params += self.linear_layer.params


    ''' for decoding, given h_,c_, y, and y_mask'''
    def Step_decoding(self):
        ## if i_label = None, then it means decoding the first word

        i_label = tensor.matrix('i_label', dtype='int64')
        i_label_mask = tensor.matrix('i_label_mask', dtype=config.floatX)
        h_ = tensor.matrix('h', dtype=config.floatX)
        c_ = tensor.matrix('c', dtype=config.floatX)
        flag = tensor.scalar('flag',dtype=config.floatX)
        state_below = tensor.alloc(numpy_floatX(0.), i_label_mask.shape[0], i_label_mask.shape[1], self.dim)
        ## shape=(1,n,d)
        shape = (i_label.shape[0],i_label.shape[1],self.dim)
        #i_label_repr = self.emb.emb_W[i_label.flatten()].reshape(shape)
        i_label_repr = self.emb.fprop(i_label)

        state_below = ifelse(tensor.gt(flag, 0.5), i_label_repr, state_below)
        if self.use_linear_layer:
            lstm_enc_y = tensor.tensor3('lstm_enc_y',dtype=config.floatX)
            state_below = tensor.concatenate((state_below,lstm_enc_y),axis=2)
            state_below = self.linear_layer.fprop(state_below)

        #state_below = tensor.switch(tensor.gt(flag, 0.5), self.emb.fprop(i_label), state_below)
        proj_h, proj_c = self.fprop(state_below,i_label_mask,h_,c_)
        proj_h, proj_c = proj_h[0],proj_c[0]

        final_layer_h = _slice(proj_h,self.layers-1,self.dim)
        proj_xx = tensor.dot(final_layer_h, self.U)
        proj_x = proj_xx + self.b
        assert proj_h.ndim == 2, 'ndim error'
        self.dbg = theano.function([i_label,i_label_mask,h_,c_,lstm_enc_y,flag],\
                    [proj_h, self.U.shape, self.b.shape],on_unused_input='ignore')
        prob = softmax(proj_x)
        if self.use_linear_layer:
            self.comp_next_probs_hc = theano.function([i_label,i_label_mask,h_,c_,lstm_enc_y,flag], \
                [tensor.shape_padleft(prob), tensor.shape_padleft(proj_h), tensor.shape_padleft(proj_c), proj_x, ])
        else:
            self.comp_next_probs_hc = theano.function([i_label,i_label_mask,h_,c_,flag], \
                [tensor.shape_padleft(prob), tensor.shape_padleft(proj_h), tensor.shape_padleft(proj_c), proj_x, ])


    ''' compute the cost (-log likelyhood) for training'''
    def Cost(self, label, y_mask, h_=None, c_=None, additional_info=None, weight=None):
        ### additional_info is the lstm encoding info of y

        enc_l = self.emb.fprop(label)## shape=(t,n,d)
        #h_ = _slice(h_c,0,4*self.dim) if h_c is not None else None
        #c_ = _slice(h_c,1,4*self.dim) if h_c is not None else None
        n_samples = label.shape[1]
        beg_label = tensor.alloc(numpy_floatX(0.), n_samples, self.dim)
        state_below = tensor.concatenate((tensor.shape_padleft(beg_label),enc_l[:-1]),axis=0)
        if additional_info is not None:
            state_below = tensor.concatenate((state_below,additional_info),axis=2)
            state_below = self.linear_layer.fprop(state_below)

        proj_h, proj_c = self.fprop(state_below,y_mask,h_,c_)

        final_layer_h = _slice(proj_h,self.layers-1,self.dim)
        self.proj_h = proj_h
        self.proj_c = proj_c
        self.final_layer_h = final_layer_h

        proj_x = tensor.dot(final_layer_h, self.U) + self.b
        assert proj_x.ndim == 3, 'ndim error'
        self.proj_x = proj_x
        pred = softmax(proj_x)
        self.pred = pred
        #self.one_step_decoder = theano.function([y,y_mask,h_,c_],[pred, self.proj_h,self.proj_c])
        self.argpred = pred.argmax(axis=2)
        fr = tensor.reshape(pred,(pred.shape[0]*pred.shape[1],pred.shape[2]))
        lr = tensor.reshape(label,(label.shape[0]*label.shape[1],))
        B = tensor.arange(fr.shape[0])
        l_pred = fr[B,lr[B]] ## it is a vector
        #l_pred = tensor.reshape(label.shape[0],label.shape[1])
        #print "y_mask.ndim",y_mask.ndim
        y_mask_r = tensor.reshape(y_mask,(y_mask.shape[0]*y_mask.shape[1],))
        assert l_pred.ndim == 1, "dimmension error"
        ### error
        cost = -tensor.log(l_pred + 1e-8)
        cost = cost * y_mask_r
        cost_matrix = tensor.reshape(cost,(y_mask.shape[0],y_mask.shape[1]))
        if weight is not None:
            cost = cost * tensor.reshape(weight,(weight.shape[0]*weight.shape[1],))
            cost_matrix = cost_matrix * weight
        sent_cost = cost_matrix.sum(axis=0)
        self.cost_matrix = cost_matrix
        cost = cost.sum()
        if self.sum_over_time:
            cost = cost /tensor.cast(y_mask.shape[1],dtype=config.floatX)
        else:
            die ## since y_mask_r contains weighted, not 0. or 1.
            cost = cost/y_mask_r.sum()
        self.dbg_list = [enc_l.shape,state_below.shape,proj_x.shape,cost,sent_cost]
        return cost, sent_cost


def get_options(mfile):
    opts = numpy.load(mfile)

    conf_opt = OrderedDict()
    conf_opt['dim_proj'] = int(opts['dim_proj'])
    conf_opt['layers'] = int(opts['layers'])
    conf_opt['n_words_x'] = int(opts['n_words_x'])
    conf_opt['n_words_y'] = int(opts['n_words_y'])
    conf_opt['reverse_src'] = bool(opts['reverse_src'])

    del opts
    return conf_opt


class EncoderDecoder(object):

    def __init__(self, config, emb_x=None, emb_y=None, emb_label=None, prefix=''):
        ### basic config
        self.config = config
        self.dim = config['dim_proj']
        self.layers = config['layers']
        self.srcvcbsize = config['n_words_x']
        self.trgvcbsize = config['n_words_y']
        self.labelvcbsize = config['n_words_label']
        self.reverse_src = config['reverse_src']
        self.reverse_trg = config['reverse_trg'] if 'reverse_trg' in config else False

        self.emb_x = emb_x if emb_x is not None else Layer_emb(self.dim,self.srcvcbsize,'src')
        self.emb_y = emb_y if emb_y is not None else Layer_emb(self.dim,self.trgvcbsize,'trg')

        if 'src_emb' in config:
            self.emb_x.reset_embd(config['src_emb'],config['srcdict'])

        if 'trg_emb' in config:
            self.emb_y.reset_embd(config['trg_emb'],config['trgdict'])

        self.emb_label = emb_label if emb_label is not None else Layer_emb(self.dim,self.labelvcbsize,'label')
        self.encoder_x = Lstm_layer(self.dim,self.layers,'enc_x')
        self.encoder_y = Lstm_layer(self.dim,self.layers,'enc_y')
        self.encoder_y_rev = Lstm_layer(self.dim,self.layers,'enc_y_rev')
        self.decoder = Lstm_decoder(self.dim,self.emb_label,self.layers,'dec',True)
        self.cost = None

        ### model params for training
        if emb_x is None:
            self.params = self.emb_x.params + self.emb_y.params + self.emb_label.params + \
                          self.encoder_x.params + self.encoder_y.params + self.decoder.params #+ self.encoder_y_rev.params 
        else:
            self.params = self.encoder_x.params + self.encoder_y.params + self.decoder.params #+ self.encoder_y_rev.params 

    def show_norm_paras(self):
        for p in self.params:
            print 'norm of ', p, ' is ', numpy.linalg.norm(p.eval())


    def unzip(self):
        new_params = OrderedDict()
        for p in self.params:
            new_params[p.name] = p.get_value()
        return new_params

    def add_dict(self, ori_dict, adder_dict):
        for kk, vv in adder_dict.iteritems():
            ori_dict[kk] = vv

    def save(self,mfile,params=None, **info):
        new_params = self.unzip() if params is None else params
        self.add_dict(new_params,info)
        self.add_dict(new_params,self.config)
        numpy.savez(mfile,**new_params)

    def zipp(self,new_params):
        for p in self.params:
            p.set_value(numpy_floatX(new_params[p.name]))

    def load(self,mfile):
        new_params = numpy.load(mfile)
        self.zipp(new_params)
        '''
        for p in self.params:
            p.set_value(new_params[p.name])
        '''

    def BuildEncDec(self):
        use_noise = theano.shared(numpy_floatX(0.))

        x = tensor.matrix('x', dtype='int64')
        y = tensor.matrix('y', dtype='int64')
        x_mask = tensor.matrix('x_mask', dtype=config.floatX)
        y_mask = tensor.matrix('y_mask', dtype=config.floatX)
        label = tensor.matrix('label', dtype='int64')
        weight = tensor.matrix('weight', dtype=config.floatX)
        self.inputs = [x,x_mask,y,y_mask,label]
        logger.debug("To Build compuatation graph")
        enc_x = self.emb_x.fprop(x)
        enc_y = self.emb_y.fprop(y)
        self.x_repr_fn = theano.function([x],enc_x)
        #self.fn_enc = theano.function(self.inputs, enc_x,on_unused_input='ignore')
        logger.debug("To Build encoding graph")
        ## the last timestep and all layers for h and c
        h_, c_ = self.encoder_x.BuildEnc(enc_x,x_mask)
        all_y_h, _ = self.encoder_y.fprop(enc_y,y_mask)
        all_y_h_rev, _ = self.encoder_y_rev.fprop(enc_y[::-1],y_mask[::-1])

        self.h_, self.c_ = h_, c_

        logger.debug("Building encoding graph over")
        logger.debug("To Build decoding graph")

        self.fn_enc = theano.function(self.inputs,c_,on_unused_input='ignore')
        self.comp_repr_enc = theano.function([x,x_mask],[self.h_,self.c_])
        self.comp_repr_enc_y = theano.function([y,y_mask],all_y_h)



        if 'weight' in self.config and self.config['weight'] > 1.0:
            self.cost, self.sent_cost = self.decoder.Cost(label,y_mask,h_,c_,all_y_h,weight)
        else:
            self.cost, self.sent_cost = self.decoder.Cost(label,y_mask,h_,c_,all_y_h)

        self.fn_proj = theano.function(self.inputs,self.decoder.proj_h,on_unused_input='ignore')
        self.fn_proj_x = theano.function(self.inputs,self.decoder.proj_x,on_unused_input='ignore')
        self.fn_prob = theano.function(self.inputs,self.decoder.pred,on_unused_input='ignore')
        self.fn_final = theano.function(self.inputs,self.decoder.final_layer_h,\
                            on_unused_input='ignore')


        logger.debug("Building decoding graph over")
        logger.debug("To Build timestep wise decoder")
        self.decoder.Step_decoding()
        logger.debug("Building timestep wise decoder over")
        #print '!!!weight',self.config['weight']
        if 'weight' in self.config and self.config['weight'] > 1.:
            self.inputs.append(weight)
        self.f_cost = theano.function(self.inputs, self.cost, name='f_cost',on_unused_input='ignore')
        self.dbg_fn = theano.function(self.inputs,[h_.shape,c_.shape,all_y_h.shape]+self.decoder.dbg_list,on_unused_input='ignore')
        self.fn_sent_cost = theano.function(self.inputs,self.sent_cost)
        self.cost_matrix_fn = theano.function(self.inputs,self.decoder.cost_matrix)
        print 'inputs',self.inputs
        return self.inputs, self.cost, use_noise


def pred_error(f_pred, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x = [data[0][t]for t in valid_index]
        y = [data[1][t] for t in valid_index]
        align = [data[2][t] for t in valid_index]
        label = [data[3][t] for t in valid_index]
        x, x_mask, y, y_mask, align, label = \
            prepare_reorderdata_minibatch(x, y, align, label)
        preds = f_pred(x, x_mask, y, y_mask)
        targets = numpy.array(y)
        valid_err += ((preds == targets)*y_mask).sum()/y_mask.sum()
        if verbose:
            print "---- batch ----"
            print "predictions == labels?"
            print preds == targets
            print "preds", preds
            print "targets", targets
            print "mask",y_mask
    valid_err = 1. - numpy_floatX(valid_err) / len(iterator)
    return valid_err

def get_weight(label,label_mask,bad_id,coeff):
    '''the label's shape is (t,n), t is the time step and n is the batch size'''
    import copy
    tag = copy.deepcopy(label).reshape(label.shape[0]*label.shape[1],)
    weight = numpy.ones(tag.shape)
    weight = weight/coeff
    for i,v in enumerate(tag):
        if v == bad_id:
            weight[i] *= coeff
    weight = weight.reshape(label.shape)
    weight = weight * label_mask
    return numpy_floatX(weight)

def construct_encdec(model_file,prefix=''):
    lstm_model = numpy.load(model_file)
    config = {}
    config['n_words_x'] = int(lstm_model['n_words_x'])
    config['n_words_y'] = int(lstm_model['n_words_y'])
    config['n_words_label'] = int(lstm_model['n_words_label'])
    config['dim_proj'] = int(lstm_model['dim_proj'])
    config['layers'] = int(lstm_model['layers'])
    config['reverse_src'] = int(lstm_model['reverse_src'])
    config['reverse_trg'] = int(lstm_model['reverse_trg']) if 'reverse_trg' in lstm_model else 0
    enc_dec = EncoderDecoder(config,prefix=prefix) 
    lstm_model = None
    enc_dec.BuildEncDec()
    enc_dec.load(model_file)

    return enc_dec


def ensemble_decoding(model_options):
    src_dict,trg_dict,label_dict = model_options['src_dict'],model_options['trg_dict'],model_options['label_dict']
    srcdict, srcdict_rev = load_dict(src_dict,True)
    assert srcdict['NULL'] == 0, 'error in srcdict'
    trgdict, trgdict_rev = load_dict(trg_dict,True)
    ldict, ldict_rev = load_dict(label_dict,True)
    beamsize=12
    enc_dec_list = []
    model_files_array = model_options['model_files'].split(',')
    for i, f in enumerate(model_files_array):
        print 'constructing model %d'%i
        enc_dec = construct_encdec(f) 
        #enc_dec = construct_encdec(f,prefix='m_%d'%i) 
        enc_dec_list.append(enc_dec)
    beamsearch = EnsembleBeamSearch(enc_dec_list,beamsize,trgdict,\
                   trgdict_rev,srcdict,srcdict_rev,ldict,ldict_rev)

    tst_file = model_options['tst_file']
    tst_ref = model_options['tst_ref']
    tst_xml = model_options['tst_xml']
    tst_l = model_options['tst_l']

    test_eval(beamsearch=beamsearch,src_file=tst_file,trg_file=tst_ref,src_xml=tst_xml,isdev=False,label_file=tst_l)

def train_lstm(model_options):
    # Model options
    src_file,trg_file,label_file = model_options['src_file'],model_options['trg_file'],model_options['label_file']
    src_dict,trg_dict,label_dict = model_options['src_dict'],model_options['trg_dict'],model_options['label_dict']
    align_file = model_options['align_file']
    reverse_src,reload_model = model_options['reverse_src'],model_options['reload_model']
    patience = model_options['patience']  # Number of epoch to wait before early stop if no progress
    max_epochs = model_options['max_epochs']  # The maximum number of epoch to run
    dispFreq = model_options['dispFreq']  # Display to stdout the training progress every N updates
    decay_c = model_options['decay_c']  # Weight decay for the classifier applied to the U weights.
    begin_valid = model_options['begin_valid']## when begin to evalute the performance on dev and test sets.
    lrate = model_options['lrate']  # Learning rate for sgd (not used for adadelta and rmsprop)
    encoder= model_options['encoder']  # TODO: can be removed must be lstm.
    saveto = model_options['saveto']  # The best model will be saved there
    validFreq = model_options['validFreq']#370,  # Compute the validation error after this number of update.
    saveFreq = model_options['saveFreq']  # Save the parameters after every saveFreq updates
    maxlen = model_options['maxlen'] # Sequence longer then this get ignored
    batch_size = model_options['batch_size'] # The batch size during training.
    valid_batch_size = model_options['valid_batch_size']# The batch size used for validation/test set.
    dataset = model_options['dataset']
    # Parameter for extra option
    noise_std = model_options['noise_std']
    use_dropout = model_options['use_dropout']#True,  # if False slightly faster, but worst test error

    print 'Loading data'
    print 'src_dict', src_dict
    if not os.path.exists(src_dict):
        construct_vcb([model_options['src_file'],model_options['dev_file'],model_options['tst_file']],src_dict)
        construct_vcb([model_options['trg_file'],model_options['dev_ref'],model_options['tst_ref']],trg_dict)
    print 'contx size',model_options['ctx_size']
    train, n_words_x, n_words_y, n_words_l = preprocess_data(src_file,trg_file,label_file,align_file,srcdict=src_dict,trgdict=trg_dict,ctx=model_options['ctx_size'])
    srcdict, srcdict_rev = load_dict(src_dict,True)
    assert srcdict['NULL'] == 0, 'error in srcdict'
    trgdict, trgdict_rev = load_dict(trg_dict,True)
    ldict, ldict_rev = load_dict(label_dict,True)
    print 'the tot number of examples is %d '%len(train[0])

    numpy.savez('train_data.npz',train)
    print "n_words_x",n_words_x
    print "n_words_y",n_words_y

    #_, valid, test = split_train(train)
    valid, test = train, train
    model_options['n_words_x'] = n_words_x
    model_options['n_words_y'] = n_words_y
    model_options['n_words_label'] = n_words_l
    if n_words_l == 1:
        model_options['n_words_label'] += 1
    print "model options", model_options

    logger.debug("Building model")
    model_options['srcdict'] = srcdict
    model_options['trgdict'] = trgdict

    enc_dec = feedforward(model_options)
    print 'initial paras'
    enc_dec.show_norm_paras()

    inputs, cost, use_noise = enc_dec.Build()
    if reload_model:
        enc_dec.load('lstm_model.npz')
        print 'use external model'
        enc_dec.show_norm_paras()

    decay_c = model_options['decay_c']
    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (enc_dec.decoder.U ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = enc_dec.f_cost#theano.function(inputs, cost, name='f_cost',on_unused_input='ignore')

    dev_file,dev_ref,dev_l,dev_align = model_options['dev_file'],model_options['dev_ref'],model_options['dev_l'],model_options['dev_align']
    tst_file,tst_ref,tst_l,tst_align = model_options['tst_file'],model_options['tst_ref'],model_options['tst_l'],model_options['tst_align']
    dev_xml,tst_xml = model_options['dev_xml'],model_options['tst_xml']

    if reload_model:
        test_eval(beamsearch=beamsearch,src_file=dev_file,trg_file=dev_ref,src_xml=dev_xml,label_file=dev_l,align_file=dev_align)
    #test_eval(beamsearch=beamsearch,src_file=tst_file,trg_file=tst_ref,src_xml=tst_xml,isdev=False,label_file=tst_l)
    print enc_dec.params
    kf_fix = get_minibatches_idx(len(train[0]), batch_size, shuffle=False)
    _, train_index = kf_fix[0]
    x = [train[0][t]for t in train_index]
    y = [train[1][t] for t in train_index]
    l = [train[2][t] for t in train_index]

    beamsize = 10
    beamsearch = BeamSearch(enc_dec,beamsize,trgdict,\
            trgdict_rev,srcdict,srcdict_rev,ldict,ldict_rev)

    print x,y,l
    xx,yy = x[0],y[0]
    x, y, label = \
        prepare_reorderdata_minibatch(x, y, l)

    print 'x',x
    print 'y',y
    print 'l',l
    print 'label'
    print label
    print 'f_cost', f_cost(x,y,l)
    print 'pred', enc_dec.f_pred(x,y)

    #weight = get_weight(label,y_mask,beamsearch.l_word2index['BAD'],4.0)
    lr = tensor.scalar(name='lr')
    trainer = Trainer()
    print inputs
    logger.debug("Compiling grads")
    f_grad_shared, f_update = trainer.SetupTainer(lr,
                                inputs, cost, enc_dec.params)
    logger.debug("Compiling grads over")

    f_update(lrate)
    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)
    print "%d train examples" % len(train[0])
    print "%d valid examples" % len(valid[0])
    print "%d test examples" % len(test[0])

    history_errs = []
    best_p = None
    bad_count = 0

    iter_epoches = len(train[0]) / batch_size
    if validFreq == -1:
        validFreq = 370 if iter_epoches < 370 else iter_epoches
    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size

    print "validFreq, saveFreq",validFreq, saveFreq
    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in xrange(max_epochs):
            tot_cost = 0
            n_samples = 0
            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)
            #kf = kf_fix[:1]
            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)
                # Select the random examples for this minibatch
                x = [train[0][t]for t in train_index]
                y = [train[1][t] for t in train_index]
                l = [train[2][t] for t in train_index]
                x, y, label = \
                    prepare_reorderdata_minibatch(x, y, l)
                n_samples += x.shape[0]
                weight,cost = None, None
                cost = f_grad_shared(x, y, label)
                f_update(lrate)
                tot_cost += cost
                ## update after testing
                #f_update(lrate)
                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected'
                    print label
                    return 1., 1., 1.
                if numpy.mod(uidx, dispFreq) == 0:
                    #, gold_prob#'cost_bak', cost_bak
                    logger.debug("Current speed is {} per update".
                            format((time.time() - start_time) / uidx))
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost
                    print '----------------'
                    beamsearch.minibatch(x,y,label)

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print 'Saving...',

                    #train_err = pred_error(f_pred, train, kf)
                    if best_p is not None:
                        params = best_p
                    else:
                        params = enc_dec.unzip()
                    enc_dec.save(saveto,params,history_errs=history_errs)
                    pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print 'Done'

            print 'Seen %d samples, tot_cost %f' % (n_samples,tot_cost)
            if eidx+1 % 5 == 0:
                sys.stdout.flush()

            if dev_file and eidx > begin_valid:
                print 'evaluating on dev and test set'
                os.system('echo iteration %d >>eval.txt'%eidx)
                test_eval(beamsearch=beamsearch,src_file=dev_file,trg_file=dev_ref,src_xml=dev_xml,label_file=dev_l,align_file=dev_align)
                if tst_file:
                    test_eval(beamsearch=beamsearch,src_file=tst_file,trg_file=tst_ref,src_xml=tst_xml,isdev=False,label_file=tst_l,align_file=tst_align)

                model_f_iter=saveto+'_iter%i.npz'%eidx
                #enc_dec.save(model_f_iter)

            if estop:
                break

    except KeyboardInterrupt:
        print "Training interupted"

    params = enc_dec.unzip()
    enc_dec.save('lstm_model.last.npz',params,history_errs=history_errs)

    #test_eval(beamsearch=beamsearch,src_file=src_file,trg_file=trg_file)
    #test_eval(src_file=src_file,trg_file=trg_file)

    decoding_check(
        enc_dec=enc_dec,
        src_file=src_file,
        trg_file=trg_file,
    )
    end_time = time.clock()
    if best_p is not None:
        enc_dec.zipp(best_p)
    else:
        best_p =  enc_dec.unzip()

    use_noise.set_value(0.)
    print_tparams(enc_dec.params)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, train, kf_train_sorted)
    valid_err = pred_error(f_pred, valid, kf_valid)
    #test_err = pred_error(f_pred, test, kf_test, True)
    test_err = pred_error(f_pred, test, kf_test)

    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err
    print "history errors", history_errs
    print 'keys of best_p', best_p.keys()

    if saveto:
        enc_dec.save(saveto,best_p,train_err=train_err,
                     valid_err=valid_err, test_err=test_err,
                     history_errs=history_errs,
        )

    print 'The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print >> sys.stderr, ('Training took %.1fs' %
                          (end_time - start_time))
    return train_err, valid_err, test_err

def get_bleu(scorefile):
    lines = open(scorefile).readlines()
    m = re.search('bleu = (.+?), length_ratio', lines[0])
    if m:
        return float(m.group(1))
    else:
        die;

def test_eval(
    beamsearch=None,
    model='lstm_model.last.npz',
    src_file='srcfile',
    trg_file='trgfile',
    label_file='lfile',
    align_file='aligfile',
    src_xml=None,
    src_dict='src.dict.pkl',
    trg_dict='trg.dict.pkl',
    l_dict='label.dict.pkl',
    beamsize=12,
    isdev=True,
):

    from itertools import izip
    import os

    res = 'src.tst.res' if not isdev else 'src.dev.res'
    ff = open(res,'w')
    os.system('mkdir -p bleudir')
    start_time = time.time()
    for i, (s,t,a) in enumerate(izip(open(src_file),open(trg_file),open(align_file))):
        _, best_tran = beamsearch.decode(s,t,a)
        if (i + 1)  % 100 == 0:
            ff.flush()
            logger.debug("Current speed is {} per sentence".
                    format((time.time() - start_time) / (i + 1)))
        print >>ff, best_tran
    ff.close()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    eval = 'python %s/../scripts/wmt15_word_level_qe_evaluation.py'%current_dir
    cmd = '%s %s %s >& eval.log; cat eval.log >>eval.txt'%(eval,label_file,res)
    eval = 'python /panfs/panmt/users/lliu/data/wmt14-qe/scripts/evaluate_wmt15.py' 
    eval = 'python ./scripts/evaluate_wmt15.py' 
    cmd = '%s %s %s %s >eval.log 2>&1 ; cat eval.log >>eval.txt'%(eval,trg_file,label_file,res)
    #cmd = '%s %s %s %s >>eval.txt'%(eval,trg_file,label_file,res)
    if isdev:
        os.system('echo dev >>eval.txt')
    else:
        os.system('echo tst >>eval.txt')

    os.system(cmd)
    print cmd
    return

    if src_xml is None:
        '''calculate bleu '''
        bleu_eval = '/panfs/panmt/users/lliu/code/mt-eval/mteval_bat_iwslt04.py'
        os.system('rm bleudir/*; cp src.res bleudir/; cp src.res src.res.bak')
        cmd = 'python %s --ref %s --workdir bleudir/'%(bleu_eval,trg_file)
        os.system(cmd)
        current_bleu = get_bleu('bleudir/src.res.score') ###
        print 'bleu on %s is %f'%(os.path.basename(src_file),current_bleu)
    else:
        toxml = "/panfs/panltg2/users/finch/expt/NEWS2015/scripts/kbest2xml.pl"
        eval = "/panfs/panltg2/users/finch/expt/NEWS2012/scripts/news_evaluation.py"
        os.system('perl %s %s src.res src.res.xml'%(toxml, src_xml))
        if isdev:
            os.system('echo dev >>eval.txt')
        else:
            os.system('echo tst >>eval.txt')

        os.system('perl %s -t %s -i src.res.xml -o eval.log >acc.log; cat acc.log; cat acc.log >>eval.txt'%(eval, trg_file))


def prob_calculater(
    beamsearch = None,
    s_seq = [],
    t_seq = [],
):
    s_seq = numpy.array(s_seq)[:,None]
    t_seq = numpy.array(t_seq)[:,None]
    s_mask = numpy.ones(s_seq.shape[0], dtype=config.floatX)[:,None]
    t_mask = numpy.ones(t_seq.shape[0], dtype=config.floatX)[:,None]
    #beamsearch.minibatch(s_seq,s_mask,t_seq,t_mask)
    return beamsearch.enc_dec.f_cost(s_seq,s_mask,t_seq,t_mask)


def decoding_check(
    enc_dec=None,
    src_file='srcfile',
    trg_file='trgfile',
    src_dict='src.dict.pkl',
    trg_dict='trg.dict.pkl',
    beamsize=12,
):
    from itertools import izip
    srcdict,srcdict_rev = load_dict(src_dict,True)
    trgdict, trgdict_rev = load_dict(trg_dict,True)
    beamsearch = BeamSearch(enc_dec,beamsize, trgdict, trgdict_rev,srcdict, srcdict_rev)

    for i, (line,t_line) in enumerate(izip(open(src_file),open(trg_file))):
        score,best_tran = beamsearch.decode(line)
        ref_score = beamsearch.prob_bisent(line,t_line)
        print "sent id %d, ref_score=%f, score=%f"%(i,ref_score,score)
        print "src: %s\ntrg: %s\nref: %s"%(line.strip(),best_tran,t_line.strip())


def init_config(
    dim_proj=50,  # word embeding dimension and LSTM number of hidden units.
    layers=1, # the number of layers for lstm encoder and decoder
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=100,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    begin_valid=0,## when begin to evalute the performance on dev and test sets.
    lrate=10,  # Learning rate for sgd (not used for adadelta and rmsprop)
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='lstm_model.npz',  # The best model will be saved there
    validFreq=-1,#370,  # Compute the validation error after this number of update.
    saveFreq=1110,  # Save the parameters after every saveFreq updates
    maxlen=100,  # Sequence longer then this get ignored
    batch_size=16,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.
    dataset='imdb',
    weight=1.0,
    # Parameter for extra option
    noise_std=0.,
    use_dropout=False,#True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
    src_file="3.zh",
    trg_file="3.en",
    label_file="3.align",
    src_dict='src.dict.pkl',
    trg_dict='trg.dict.pkl',
    label_dict='label.dict.pkl',
    dev_file='',
    dev_ref='',
    dev_xml='',
    dev_l='',
    tst_l='',
    tst_file='',
    tst_ref='',
    tst_xml='',
    reverse_src=True,
):
    return locals().copy()

def add_dir(filename,path):
    if '/' in filename:
        return filename
    else:
        return path+'/'+filename

def get_train_opt(file):
    config = init_config()
    for line in open(file):
        fields = line.strip().split()
        if len(fields) < 3:
            continue
        if fields[0] != 'int' and fields[0] != 'str' and fields[0] != 'float':
            continue
        type, k, v = fields[0], fields[1], fields[2]
        config[k] = eval(type)(v)
        print type, k, v
    assert config["src_file"] is not None, 'reset src_file'

    if 'data_dir' in config:
        dir = config['data_dir']

        files = ['src_file','trg_file','label_file','align_file'] + \
                ['dev_file','dev_ref','dev_l','dev_align'] + \
                ['tst_file','tst_ref','tst_l','tst_align']
        for x in files:
            config[x] = add_dir(config[x],dir)

        '''
        config['src_file'] = '%s/%s'%(dir,config['src_file'])
        config['trg_file'] = '%s/%s'%(dir,config['trg_file'])
        config['label_file'] = '%s/%s'%(dir,config['label_file'])
        config['align_file'] = '%s/%s'%(dir,config['align_file'])

        config['dev_file'] = '%s/%s'%(dir,config['dev_file'])
        config['dev_ref'] = '%s/%s'%(dir,config['dev_ref'])
        config['dev_l'] = '%s/%s'%(dir,config['dev_l'])
        config['dev_align'] = '%s/%s'%(dir,config['dev_align'])

        config['tst_file'] = '%s/%s'%(dir,config['tst_file'])
        config['tst_ref'] = '%s/%s'%(dir,config['tst_ref'])
        config['tst_l'] = '%s/%s'%(dir,config['tst_l'])
        config['tst_align'] = '%s/%s'%(dir,config['tst_align'])
        '''
        del config['data_dir']

    if 'model_files' in config:
        ## ensemble decoding
        ensemble_decoding(config)
    else:
        train_lstm(config)


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG,
        format="%(asctime)s: %(filename)s:%(lineno)s - %(funcName)s: %(levelname)s: %(message)s")

    dir = "/panfs/panmt/users/lliu/code/lstm_reorder/h10k-iwslt-data"
    src = "%s/h10k.zh"%dir
    trg = "%s/h10k.en"%dir
    align = "%s/h10k.align"%dir


    get_train_opt(sys.argv[1])
    sys.exit() 

    model_options = init_config()
    model_options['src'] = 'train.source'
    model_options['trg'] = 'train.target'
    model_options['label'] = 'train.tags'
    model_options['dev_src'] = 'data/dev/dev.source'
    model_options['dev_ref'] = 'data/dev/dev.target'
    model_options['dev_l'] = 'data/dev/dev.tags'
    model_options['tst_src'] = 'data/test/test.source'
    model_options['tst_ref'] = 'data/test/test.target'
    model_options['tst_l'] = 'data/test/test.tags'

    train_lstm(model_options)
