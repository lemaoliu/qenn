import cPickle
import gzip
import os
import sys
from collections import defaultdict

import numpy
import theano


def pkl2txt(path):

    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    train_set = cPickle.load(f)
    f.close()

    def save(xlist, filename):
        ff = open(filename,'w')
        for x in xlist: 
            print >>ff, x
        ff.close()

    save(train_set[0],"train_x.txt")
    save(train_set[1],"train_y.txt")



def get_label(align,distortion_limit=10):

    ## relative distortion: -dl, -dl+1, ..., -1, 0, 1, ..., dl, NULL
    ## its corresponding class: 0,      1,  ..., -1+dl, dl, dl+1, ..., 2*dl, 2*dl+1
    num_classes = 2*distortion_limit + 2
    label = [0]*len(align)
    pre_null_p, cur_p = -1, 0
    while cur_p < len(align):
        if align[cur_p] != -1:
            if cur_p == 0:
                ## translate the first src word
                label[cur_p] = align[cur_p] + distortion_limit + 1 #if align[cur_p] >0 \
                    #else distortion_limit + 1 
            elif pre_null_p != -1:
                label[cur_p] = align[cur_p] - align[pre_null_p] + distortion_limit
            else: # cur_p>0 and pre_null_p == -1
                ## translate the first src word after inserting several words previously
                label[cur_p] = align[cur_p] + distortion_limit + 1#if align[cur_p] >0 \
                    #else distortion_limit + 1          
            pre_null_p = cur_p

            label[cur_p] = min(2*distortion_limit,label[cur_p])# or label[cur_p] < 0:
            label[cur_p] = max(0,label[cur_p])# or label[cur_p] < 0:
        else:
            label[cur_p] = num_classes-1        
        cur_p +=1        
    '''
    print align
    print label
    print zip(align,label)
    '''
    return label

def prepare_reorderdata_minibatch(seqs_x, seqs_y, seqs_label, maxlen=None):
    """Create the matrices from the datasets.
    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.
    if maxlen is set, we will cut all sequence to this maximum
    lenght.
    This swap the axis!
    """
    # x: a list of sentences

    x = numpy.array(seqs_x)
    y = numpy.array(seqs_y)
    labels = numpy.array(seqs_label)
    '''
    if not numpy.array_equal(x, y):
        print numpy.array(seqs_x).shape
        print numpy.array(seqs_y).shape
        print numpy.array(seqs_label).shape
    '''
    return x, y, labels

def split_train(train_set,maxlen=None,
              valid_test_portion=0.2,
              sort_by_len=True):

    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        new_train_set_a = []
        new_train_set_l = []
        for x, y, a,l in zip(train_set[0], train_set[1],train_set[2],train_set[3]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
                new_train_set_a.append(a)
                new_train_set_l.append(l)
        train_set = (new_train_set_x, new_train_set_y,new_train_set_a,new_train_set_l)
        del new_train_set_x, new_train_set_y, new_train_set_a, new_train_set_l

    # split training set into validation set
    train_set_x, train_set_y, train_set_a, train_set_l = train_set
    n_samples = len(train_set_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_test_portion)))

    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    valid_set_a = [train_set_a[s] for s in sidx[n_train:]]
    valid_set_l = [train_set_l[s] for s in sidx[n_train:]]

    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    train_set_a = [train_set_a[s] for s in sidx[:n_train]]
    train_set_l = [train_set_l[s] for s in sidx[:n_train]]

    n_valid = int(numpy.round(len(valid_set_x) * 0.5))
    test_set_x = valid_set_x[:n_valid]
    test_set_y = valid_set_y[:n_valid]
    test_set_a = valid_set_a[:n_valid]
    test_set_l = valid_set_l[:n_valid]

    valid_set_x = valid_set_x[n_valid:]
    valid_set_y = valid_set_y[n_valid:]
    valid_set_a = valid_set_a[n_valid:]
    valid_set_l = valid_set_l[n_valid:]

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]
        test_set_a = [test_set_a[i] for i in sorted_index]
        test_set_l = [test_set_l[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]
        valid_set_a = [valid_set_a[i] for i in sorted_index]
        valid_set_l = [valid_set_l[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]
        train_set_a = [train_set_a[i] for i in sorted_index]
        train_set_l = [train_set_l[i] for i in sorted_index]

    train = (train_set_x, train_set_y, train_set_a, train_set_l)
    valid = (valid_set_x, valid_set_y, valid_set_a, valid_set_l)
    test = (test_set_x, test_set_y, test_set_a, test_set_l)

    return train, valid, test

def load_dict(dictfilename, bidirection=False):

    npdict = cPickle.load(open(dictfilename,'rb'))
    word_dict = defaultdict(int)
    for k,v in npdict.iteritems():
        word_dict[str(k)] = int(v)
    if not bidirection:
        return word_dict
    else:
        dict_rev = defaultdict()
        for k,v in npdict.iteritems():
            dict_rev[int(v)] = str(k) #'file' if str(k) == '$filefile$' else str(k)
        return word_dict, dict_rev

def load(filename,dict=None):
    word_dict = defaultdict(int) if dict is None else load_dict(dict) 
    tmpid = 3 ## if it is not 2, for example 3, it may crash when look up the key 3.
    train_set=[]
    if dict is not None:
        assert 'NULL' not in word_dict or word_dict['NULL'] == 0, 'error in dict'
    else:
        word_dict['NULL'] = 0 
        word_dict['BOS'] = 1
        word_dict['EOS'] = 2

    for line in open(filename,'r'):
        words = line.strip().split()
        line_ids = []
        for w in words:
            #if w == 'file': w = '$filefile$'### required by numpy.savez
            if word_dict[w] == 0 and w != 'NULL':
                if dict is None:### construct the word_dict
                    word_dict[w] = tmpid
                    tmpid += 1
                else:
                    word_dict[w] = word_dict['NULL']
            line_ids.append(word_dict[w])
        train_set.append(line_ids)
    assert word_dict['NULL'] == 0, 'error in srcdict NULL ID %d'%word_dict['NULL']
    return train_set,word_dict

def construct_vcb(filenames,saveto):
    ###  construct the vcb from a list of filenames
    import sys
    names = ' '.join(filenames)
    print 'concatenate files:', names
    os.system('cat %s >allfile'%names)
    _, vcb = load('allfile')
    print '%s vcb size is %d'%(saveto,len(vcb.keys()))
    cPickle.dump(vcb,open(saveto,'wb')) 


def seq2indices(words,word_dict):
    line_ids = []
    for w in words:
        #if w == 'file': w = '$filefile$'### required by numpy.savez
        line_ids.append(word_dict[w])      
    return line_ids

def get_align_map(align_line):
    aligns = align_line.split()
    t_align_dict=defaultdict(list)
    for a in aligns:
        s,t = map(int,a.split('-'))
        t_align_dict[t].append(s)
    return t_align_dict

def get_src_pos(cur,max_len,align_map):
    cands = [cur]
    for d in xrange(1, max_len+1):
        i = cur - d
        if i >= 0 and i < max_len:
            cands.append(i)
        i = cur + d
        if i >= 0 and i < max_len:
            cands.append(i)
    for x in cands:
        fert = align_map[x]
        if len(fert) == 1:
            return fert[0]
        elif len(fert) > 1:
            fert = sorted(fert)
            return fert[len(fert)/2] ## return mid
    return None

def get_ctx(cur,src,ctx_size):
    mid_size = ctx_size/2
    assert ctx_size%2, 'error'
    b_s = max(cur-mid_size,0)
    e_s = min(cur+mid_size+1,len(src))
    ctx = []
    if cur-mid_size < 0:
        ctx += [1]*(mid_size-cur)
    ctx += src[b_s:e_s]
    if cur+mid_size+1 > len(src):
        ctx += [2]*(cur+mid_size+1-len(src))
    assert len(ctx) == ctx_size, 'size error'
    return ctx

def generate_instances_nolabel(src,trg,align_line,ctx_size):
    '''src, trg and label are lists of word id; align_line is a string'''
    max_len = len(trg) 
    align_map = get_align_map(align_line)
    instances = []
    for i_t, t in enumerate(trg):
        i_s = get_src_pos(i_t,max_len,align_map)
        if i_s is None:
            print src,trg,align_line
        x = get_ctx(i_s,src,ctx_size)
        y = get_ctx(i_t,trg,ctx_size)
        instances.append((x,y,i_t))
    return instances

def generate_instances(src,trg,label,align_line,ctx_size):
    '''src, trg and label are lists of word id; align_line is a string'''
    #print 'src',src
    #print 'trg',trg
    #print 'align',align_line
    max_len = len(trg) 
    align_map = get_align_map(align_line)
    instances = []
    for i_t, t in enumerate(trg):
        i_s = get_src_pos(i_t,max_len,align_map)
        if i_s is None:
            print i_t,src,trg,align_line,label
        x = get_ctx(i_s,src,ctx_size)
        #print 'i_s ctx',i_s,x 
        #die
        y = get_ctx(i_t,trg,ctx_size)
        l = label[i_t]
        ## x and y are lists, but l is int
        instances.append((x,y,l))
        #print 'src ctx',x, 'trg ctx',y, 'i_t',i_t, 'i_s', i_s
    return instances




def preprocess_data(src,trg,label,align,srcdict=None,trgdict=None,ldict=None,ctx=3):
    from collections import defaultdict
    ## do not use null and eos for label, since label is binary for word-level quality estimation
    def load_label(filename,dict=None):
        word_dict = {} if dict is None else load_dict(dict) 
        word_dict['OK'] = 0
        word_dict['BAD'] = 1
        tmpid = 2 
        train_set=[]
        for line in open(filename,'r'):
            words = line.strip().split()
            line_ids = []
            for w in words:
                #if w == 'file': w = '$filefile$'### required by numpy.savez
                if w not in word_dict:
                    if dict is None:### construct the word_dict
                        word_dict[w] = tmpid
                        tmpid += 1
                    else:
                        die
                        word_dict[w] = word_dict['NULL']
                line_ids.append(word_dict[w])      
            train_set.append(line_ids)
        if dict is not None: assert tmpid == 0, 'error in frozen dict'
        return train_set,word_dict

    src_set, src_dict = load(src,srcdict) 
    assert src_dict['NULL'] == 0, 'error in srcdict NULL ID %d'%src_dict['NULL']
    trg_set, trg_dict = load(trg,trgdict)
    label_set, label_dict = load_label(label,ldict)## no null and eos

    if srcdict is None:
        cPickle.dump(src_dict,open('src.dict.pkl','wb')) 
    if trgdict is None:
        cPickle.dump(trg_dict,open('trg.dict.pkl','wb')) 
    if ldict is None:
        cPickle.dump(label_dict,open('label.dict.pkl','wb')) 

    train_x,train_y,train_l = [],[],[]
    for i, a in enumerate(open(align)):
        x,y,l = src_set[i],trg_set[i],label_set[i]
        instances = generate_instances(x,y,l,a,ctx)
        for a,b,c in instances:
            train_x.append(a)
            train_y.append(b)
            train_l.append(c)

    assert src_dict['NULL'] == 0, 'error in srcdict'

    return (train_x,train_y,train_l), \
            numpy.max(src_dict.values())+1, numpy.max(trg_dict.values())+1, \
            numpy.max(label_dict.values())+1


def prepare_data(seqs, labels, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels


def get_dataset_file(dataset, default_dataset, origin):
    '''Look for it as if it was a full path, if not, try local file,
    if not try in the data directory.

    Download dataset if it is not present

    '''
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == default_dataset:
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == default_dataset:
        import urllib
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)
    return dataset


def load_data(path="imdb.pkl", n_words=100000, valid_portion=0.1, maxlen=None,
              sort_by_len=True):
    '''Loads the dataset

    :type path: String
    :param path: The path to the dataset (here IMDB)
    :type n_words: int
    :param n_words: The number of word to keep in the vocabulary.
        All extra words are set to unknow (1).
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.

    '''

    #############
    # LOAD DATA #
    #############

    # Load the dataset
    path = get_dataset_file(
        path, "imdb.pkl",
        "http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl")

    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    train_set = cPickle.load(f)
    test_set = cPickle.load(f)
    f.close()
    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

    # split training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)

    return train, valid, test
