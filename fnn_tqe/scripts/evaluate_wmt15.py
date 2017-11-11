from __future__ import division, print_function

import numpy as np
from argparse import ArgumentParser
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('experiment_logger')

#------------------ evaluation for the WMT15 format:------------------------
#
#       <METHOD NAME> <SEGMENT NUMBER> <WORD INDEX> <WORD> <BINARY SCORE>
#
#---------------------------------------------------------------------------


#-------------PREPROCESSING----------------
# check if <a_list> is a list of lists
def list_of_lists(a_list):
    if isinstance(a_list, (list, tuple, np.ndarray)) and len(a_list) > 0 and all([isinstance(l, (list, tuple, np.ndarray)) for l in a_list]):
        return True
    return False


# check that two lists of sequences have the same number of elements
def check_word_tag(words_seq, tags_seq, dataset_name=''):
    assert(len(words_seq) == len(tags_seq)), "Number of word and tag sequences doesn't match in dataset {}".format(dataset_name)
    for idx, (words, tags) in enumerate(zip(words_seq, tags_seq)):
        assert(len(words) == len(tags)), "Numbers of words and tags don't match in sequence {} of dataset {}".format(idx, dataset_name)


def parse_submission(ref_txt_file, ref_tags_file, submission):
    tag_map = {'OK': 1, 'BAD': 0}
    # parse test tags
    true_words = []
    for line in open(ref_txt_file):
        true_words.append(line[:-1].decode('utf-8').split())

    # parse test txt
    true_tags = []
    for line in open(ref_tags_file):
        true_tags.append([tag_map[t] for t in line[:-1].decode('utf-8').split()])
    check_word_tag(true_words, true_tags, dataset_name='reference')

    # parse and check the submission
    test_tags = []
    for line in open(submission):
        test_tags.append([tag_map[t] for t in line[:-1].decode('utf-8').split()])
    # test_tags = [[] for i in range(len(true_tags))]
    # for idx, line in enumerate(open(submission)):
    #     chunks = line[:-1].decode('utf-8').strip('\r').split('\t')
    #     test_tags[int(chunks[1])].append(tag_map[chunks[4]])
    check_word_tag(true_words, test_tags, dataset_name='prediction')

    return true_tags, test_tags


#---------------------SEQUENCE CORRELATION-------------------
# each span is a pair (span start, span end). span end = last span element + 1
def get_spans(sentence, good_label=1, bad_label=0):
    good_spans, bad_spans = [], []
    prev_label = None
    cur_start = 0
    for idx, label in enumerate(sentence):
        if label == good_label:
            if label != prev_label:
                if prev_label is not None:
                    bad_spans.append((cur_start, idx))
                cur_start = idx
        elif label == bad_label:
            if label != prev_label:
                if prev_label is not None:
                    good_spans.append((cur_start, idx))
                cur_start = idx
        else:
            print("Unknown label", label)
        prev_label = label
    # add last span
    if prev_label == good_label:
        good_spans.append((cur_start, len(sentence)))
    else:
        bad_spans.append((cur_start, len(sentence)))
    return(good_spans, bad_spans)


def intersect_spans(true_span, pred_span):
    # connectivity matrix for all pairs of spans from the reference and prediction
    connections = [[max(0, min(t_end, p_end) - max(t_start, p_start)) for (p_start, p_end) in pred_span] for (t_start, t_end) in true_span]
    adjacency = np.array(connections)
    res = 0
    # while there are non-zero elements == there are unused spans
    while adjacency.any():
        # maximum intersection
        max_el = adjacency.max()
        max_coord = adjacency.argmax()
        # coordinates of the max element
        coord_x, coord_y = max_coord // adjacency.shape[1], max_coord % adjacency.shape[1]
        res += max_el

        # remove all conflicting edges
        for i in range(adjacency.shape[0]):
            adjacency[i][coord_y] = 0
        for i in range(adjacency.shape[1]):
            adjacency[coord_x][i] = 0
    return res


def sequence_correlation_weighted(y_true, y_pred, good_label=1, bad_label=0, out='sequence_corr.out', verbose=False):
    assert(len(y_true) == len(y_pred))
    if not list_of_lists(y_true) and not list_of_lists(y_pred):
        logger.warning("You provided the labels in a flat list of length {}. Assuming them to be one sequence".format(len(y_true)))
        y_true = [y_true]
        y_pred = [y_pred]
    elif list_of_lists(y_true) and list_of_lists(y_pred):
        pass
    else:
        logger.error("Shapes of the hypothesis and the reference don't match")
        return 0

    sentence_pred = []
    if verbose:
        out_file = open(out, 'w')
    for true_sent, pred_sent in zip(y_true, y_pred):
        ref_bad = sum([1 for l in true_sent if l == bad_label])
        ref_good = sum([1 for l in true_sent if l == good_label])
        assert(ref_bad + ref_good == len(true_sent))
        # coefficients that ensure the equal influence of good and bad classes on the overall score
        try:
            coeff_bad = len(true_sent)/(2*ref_bad)
        except ZeroDivisionError:
            coeff_bad = 0.0
        try:
            coeff_good = len(true_sent)/(2*ref_good)
        except ZeroDivisionError:
            coeff_good = 0.0

        assert(len(true_sent) == len(pred_sent))
        true_spans_1, true_spans_0 = get_spans(true_sent, good_label=good_label, bad_label=bad_label)
        pred_spans_1, pred_spans_0 = get_spans(pred_sent, good_label=good_label, bad_label=bad_label)

        res_1 = intersect_spans(true_spans_1, pred_spans_1)
        res_0 = intersect_spans(true_spans_0, pred_spans_0)

        len_t_1, len_t_0 = len(true_spans_1), len(true_spans_0)
        len_p_1, len_p_0 = len(pred_spans_1), len(pred_spans_0)
        if len_t_1 + len_t_0 > len_p_1 + len_p_0:
            spans_ratio = (len_p_1 + len_p_0)/(len_t_1 + len_t_0)
        else:
            spans_ratio = (len_t_1 + len_t_0)/(len_p_1 + len_p_0)

        corr_val = (res_1*coeff_good + res_0*coeff_bad)*spans_ratio/float(len(true_sent))
        if verbose:
            out_file.write("Reference:  %s\nPrediction: %s\nCorrelation: %s\n" % (' '.join([str(t) for t in true_sent]), ' '.join([str(t) for t in pred_sent]), str(corr_val)))
        sentence_pred.append(corr_val)

    if verbose:
        out_file.close()
    return sentence_pred, np.average(sentence_pred)


#---------------------------EVALUATION-------------------------
# convert list of lists into a flat list
def flatten(lofl):
    if list_of_lists(lofl):
        return [item for sublist in lofl for item in sublist]
    elif type(lofl) == dict:
        return lofl.values()


def compute_scores(true_tags, test_tags, filename, seq_corr_file=None):
    flat_true = flatten(true_tags)
    flat_pred = flatten(test_tags)
    print("---------------{}-----------------".format(filename))
    print("F1 score for classes: ", f1_score(flat_true, flat_pred, average=None, pos_label=None))
    #print("Precision for classes: ", precision_score(flat_true, flat_pred, average=None, pos_label=None))
    #print("Recall for classes: ", recall_score(flat_true, flat_pred, average=None, pos_label=None))
    print("Average F1 score: ", f1_score(flat_true, flat_pred, average='weighted', pos_label=None))

    seq_corr = sequence_correlation_weighted(true_tags, test_tags)
    if seq_corr_file is not None:
        seq_corr_out = open(seq_corr_file, 'w')
        for sc in seq_corr[0]:
            seq_corr_out.write("%f\n" % sc)
    print("Sequence correlation: {}".format(seq_corr[1]))


def evaluate(ref_txt_file, ref_tags_file, submission, tmp_dir, seq_corr_file=None):
    true_tags, test_tags = parse_submission(ref_txt_file, ref_tags_file, submission)
    compute_scores(true_tags, test_tags, submission, seq_corr_file=seq_corr_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("ref_txt", action="store", help="test target text (one line per sentence)")
    parser.add_argument("ref_tags", action="store", help="test tags (one line per sentence)")
    parser.add_argument("submission", action="store", help="submission (wmt15 format)")
    parser.add_argument("--tmp", default="tmp", help="folder to store the data generated by the script (default $PWD/tmp)")
    parser.add_argument("--sequence", dest='seq_corr', default=None, help="file to store sentence-level sequence correlation scores (not saved if no file provided)")
    args = parser.parse_args()

    evaluate(args.ref_txt, args.ref_tags, args.submission, args.tmp, seq_corr_file=args.seq_corr)
