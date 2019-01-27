import torch


def get_gt_ranks(ranks, ans_ind):
    ans_ind = ans_ind.view(-1)
    gt_ranks = torch.LongTensor(ans_ind.size(0)) # Looks like this is shaped [q1r1, q2r1, ..., q1r2,]
    for i in range(ans_ind.size(0)):
        gt_ranks[i] = int(ranks[i, ans_ind[i]])
    return gt_ranks


def process_ranks(ranks):
    num_ques = ranks.size(0)
    num_opts = 100

    # none of the values should be 0, there is gt in options
    if torch.sum(ranks.le(0)) > 0:
        num_zero = torch.sum(ranks.le(0))
        print("Warning: some of ranks are zero: {}".format(num_zero))
        ranks = ranks[ranks.gt(0)]

    # rank should not exceed the number of options
    if torch.sum(ranks.ge(num_opts + 1)) > 0:
        num_ge = torch.sum(ranks.ge(num_opts + 1))
        print("Warning: some of ranks > 100: {}".format(num_ge))
        ranks = ranks[ranks.le(num_opts + 1)]

    ranks = ranks.float()
    num_r1 = float(torch.sum(torch.le(ranks, 1)))
    num_r5 = float(torch.sum(torch.le(ranks, 5)))
    num_r10 = float(torch.sum(torch.le(ranks, 10)))
    print("\tNo. questions: {}".format(num_ques))
    print("\tr@1: {}".format(num_r1 / num_ques))
    print("\tr@5: {}".format(num_r5 / num_ques))
    print("\tr@10: {}".format(num_r10 / num_ques))
    print("\tmeanR: {}".format(torch.mean(ranks)))
    print("\tmeanRR: {}".format(torch.mean(ranks.reciprocal())))

# We want to rank the data - i.e. predicted rank of item at proper index
def scores_to_ranks(scores):
    # sort in descending order - largest score gets highest rank
    # Shapes: 5x10x100 (trial x round x response)
    # return scores.argsort()[::-1] + 1 # negative strides not supported
    # https://stackoverflow.com/questions/5284646/rank-items-in-an-array-using-python-numpy
    order = (-1 * scores).argsort()
    ranks = order.argsort() + 1

    # Flatten trials and rounds
    # return ranks.permute(1, 0, 2).contiguous().view(-1, 100)
    # Because batch is the first dimension, if we flatten here, it will list out by trial first (q1r1, q2r1, etc...)
    # However ans_ind is composed by concatenation of dialog answers, meaning (q1r1, q1r2, etc...)
    # Thus we need an axis swap
    # sorted_ranks, ranked_idx = scores.sort(1, descending=True)
    # convert from ranked_idx to ranks
    # ranks = ranked_idx.clone().fill_(0)
    # for i in range(ranked_idx.size(0)):
        # for j in range(100):
            # ranks[i][ranked_idx[i][j]] = j
    # ranks += 1
    # return ranks
