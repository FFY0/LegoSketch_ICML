import numpy as np

# space_budget x KB
def CM_sketch(labels, space_budget, hash_num,top_ratio=1):
    print('CMsketch:',space_budget)
    n_buckets = int(space_budget * 1024 / (hash_num * 4))
    # ARE, AAE = count_min(labels, n_buckets, hash_num,top_ratio=top_ratio)
    # return ARE, AAE


# space_budget x KB
def C_sketch(labels, space_budget, hash_num,top_ratio=1):
    print('Csketch:',space_budget)
    n_buckets = int(space_budget * 1024 / (hash_num * 4))
    # ARE, AAE = count_sketch(labels, n_buckets, hash_num,top_ratio=top_ratio)
    # return ARE, AAE



def random_hash(y, n_buckets):
    """ Sketch with a random hash
    Args:
        y: true counts of each item, float - [num_items]
        n_buckets: number of buckets

    Returns
        counts: estimated counts in each bucket, float - [num_buckets]
        loss: estimation error
        y_bueckets: item -> bucket mapping - [num_items]
    """
    counts = np.zeros(n_buckets)
    y_buckets = np.random.choice(np.arange(n_buckets), size=len(y))
    for i in range(len(y)):
        counts[y_buckets[i]] += y[i]

    # loss = compute_avg_loss(counts, y, y_buckets)
    return counts, '', y_buckets

def learned_count_sketch_without_pred(y,total_buckets):
    # get the num of items
    n = len(y)
    # T = int(np.log2(np.log2(n)))
    T = 10
    # threshold = T / total_buckets
    threshold = 1 * (n/(total_buckets/3))
    # put all items into the T sub count-sketch
    n_hash = 3
    sub_n_bucket = int(total_buckets // (6*T))
    T_sub_pred = []
    for t in range(T):
        sub_pred = count_sketch(y, sub_n_bucket, n_hash)
        T_sub_pred.append(sub_pred)
    main_n_bucket = (total_buckets - (sub_n_bucket * T*3))//3
    main_pred = count_sketch(y, main_n_bucket, n_hash)    
    T_sub_pred = np.stack(T_sub_pred, axis=0)
    median_sub_pred = np.median(T_sub_pred, axis=0)
    flag = median_sub_pred >= threshold
    final_pred = np.where(flag, main_pred, 0)
    return final_pred

def count_min(y, n_buckets, n_hash):
    """ Count-Min
    Args:
        y: true counts of each item, float - [num_items]
        n_buckets: number of buckets
        n_hash: number of hash functions

    Returns:
        Estimation error
    """
    if len(y) == 0:
        return 0  # avoid division of 0

    counts_all = np.zeros((n_hash, n_buckets))
    y_buckets_all = np.zeros((n_hash, len(y)), dtype=int)
    for i in range(n_hash):
        counts, _, y_buckets = random_hash(y, n_buckets)
        counts_all[i] = counts
        y_buckets_all[i] = y_buckets
    y_est = np.zeros(len(y))
    for i in range(len(y)):
        est = np.min([counts_all[k, y_buckets_all[k, i]] for k in range(n_hash)])
        y_est[i] = est
    return y_est



def count_sketch(y, n_buckets, n_hash):
    """ Count-Sketch
    Args:
        y: true counts of each item, float - [num_items]
        n_buckets: number of buckets
        n_hash: number of hash functions

    Returns:
        Estimation error
    """
    if len(y) == 0:
        return 0  # avoid division of 0
    counts_all = np.zeros((n_hash, n_buckets))
    y_buckets_all = np.zeros((n_hash, len(y)), dtype=int)
    y_signs_all = np.zeros((n_hash, len(y)), dtype=int)
    for i in range(n_hash):
        counts, y_buckets, y_signs = random_hash_with_sign(y, n_buckets)
        counts_all[i] = counts
        y_buckets_all[i] = y_buckets
        y_signs_all[i] = y_signs
    y_est = np.zeros(len(y))
    for i in range(len(y)):
        est = np.median(
            [y_signs_all[k, i] * counts_all[k, y_buckets_all[k, i]] for k in range(n_hash)])
        y_est[i] = est
    return y_est

def random_hash_with_sign(y, n_buckets):
    """ Assign items in y into n_buckets, randomly pick a sign for each item
    Args:
        y: true counts of each item, float - [num_items]
        n_buckets: number of buckets

    Returns
        counts: estimated counts in each bucket, float - [num_buckets]
        loss: estimation error
        y_bueckets: item -> bucket mapping - [num_items]
    """
    counts = np.zeros(n_buckets)
    y_buckets = np.random.choice(np.arange(n_buckets), size=len(y))
    y_signs = np.random.choice([-1, 1], size=len(y))
    for i in range(len(y)):
        counts[y_buckets[i]] += (y[i] * y_signs[i])
    return counts, y_buckets, y_signs

