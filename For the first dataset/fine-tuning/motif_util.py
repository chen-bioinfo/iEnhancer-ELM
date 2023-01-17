#### ::: utils for DNABERT-viz motif search ::: ####

import os
import pandas as pd
import numpy as np

def kmer2seq(kmers):
    """
    Convert kmers to original sequence
    
    Arguments:
    kmers -- str, kmers separated by space.
    
    Returns:
    seq -- str, original sequence.
    """
    kmers_list = kmers.split(" ")
    bases = [kmer[0] for kmer in kmers_list[0:-1]]
    bases.append(kmers_list[-1])
    seq = "".join(bases)
    assert len(seq) == len(kmers_list) + len(kmers_list[0]) - 1
    return seq

def seq2kmer(seq, k):
    """
    Convert original sequence to kmers
    
    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.
    
    Returns:
    kmers -- str, kmers separated by space
    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers

def delete_gap(idx, max_gap=0, ignore_gap=False):
    """
    @author: lijiahao
    Function:
        to overcome the issue that the gap between two continual attention motif 
        is less that the max_gap. if happens, we should ignore the gap and connect them
    
    Keyword argument:
    idx:
        the indexes of the attention motif
    max_gap:
        the max gap that we should ignore.
    
    Return:
        a idx, which ignore the gap
    """

    if ignore_gap == False: # no ignore the gap
        return idx
    
    merge_index = []
    for i in range(len(idx)-1):
        curr_end = idx[i][1]
        next_start = idx[i+1][0]
        if next_start - curr_end <= max_gap:
            merge_index.append(i)
    
    for i in range(len(merge_index)-1, 0-1, -1):
        index = merge_index[i]
        idx[index][1] = idx[index+1][1]
        idx = np.delete(idx, index+1, axis=0)
      
    return idx

def contiguous_regions(condition, len_thres=5, max_gap=0):
    """
    Modified from and credit to: https://stackoverflow.com/a/4495197/3751373
    Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index.
    Arguments:
    condition -- custom conditions to filter/select high attention 
            (list of boolean arrays)
    
    Keyword arguments:
    len_thres -- int, specified minimum length threshold for contiguous region 
        (default 5)
    Returns:
    idx -- Index of contiguous regions in sequence
    """
    
    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    
    idx = delete_gap(idx, max_gap, True)

    # eliminate those not satisfying length of threshold
    idx = idx[np.argwhere((idx[:,1]-idx[:,0])>=len_thres).flatten()]
    return idx

def find_high_attention(score, min_len=5, max_gap=0, **kwargs):
    """
    With an array of attention scores as input, finds contiguous high attention 
    sub-regions indices having length greater than min_len.
    
    Arguments:
    score -- numpy array of attention scores for a sequence
    Keyword arguments:
    min_len -- int, specified minimum length threshold for contiguous region 
        (default 5)
    **kwargs -- other input arguments:
        cond -- custom conditions to filter/select high attention 
            (list of boolean arrays)
    
    Returns:
    motif_regions -- indices of high attention regions in sequence
    """
    
    cond1 = (score > np.mean(score))
    cond2 = (score > 1*np.min(score))
    cond = [cond1, cond2]
    
    cond = list(map(all, zip(*cond)))
    
    if 'cond' in kwargs: # if input custom conditions, use them
        cond = kwargs['cond']
        if any(isinstance(x, list) for x in cond): # if input contains multiple conditions
            cond = list(map(all, zip(*cond)))
    
    cond = np.asarray(cond)
        
    # find important contiguous region with high attention
    motif_regions = contiguous_regions(cond,min_len, max_gap=max_gap)
    
    return motif_regions

def count_motif_instances(seqs, motifs, allow_multi_match=False):
    """
    Use Aho-Corasick algorithm for efficient multi-pattern matching
    between input sequences and motif patterns to obtain counts of instances.
    
    Arguments:
    seqs -- list, numpy array or pandas series of DNA sequences
    motifs -- list, numpy array or pandas series, a collection of motif patterns
        to be matched to seqs
    Keyword arguments:
    allow_multi_match -- bool, whether to allow for counting multiple matchs (default False)
    Returns:
    motif_count -- count of motif instances (int)
    
    """
    import ahocorasick 
    from operator import itemgetter
    
    motif_count = {}
    
    A = ahocorasick.Automaton()
    for idx, key in enumerate(motifs):
        A.add_word(key, (idx, key))
        motif_count[key] = 0
    A.make_automaton()
    
    for seq in seqs:
        matches = sorted(map(itemgetter(1), A.iter(seq)))
        matched_seqs = []
        for match in matches:
            match_seq = match[1]
            assert match_seq in motifs
            if allow_multi_match:
                motif_count[match_seq] += 1
            else: # for a particular seq, count only once if multiple matches were found
                if match_seq not in matched_seqs:
                    motif_count[match_seq] += 1
                    matched_seqs.append(match_seq)
    
    return motif_count

def motifs_hypergeom_test(pos_seqs, neg_seqs, motifs, p_adjust = 'fdr_bh', alpha = 0.05, verbose=False, 
                          allow_multi_match=False, **kwargs):
    """
    Perform hypergeometric test to find significantly enriched motifs in positive sequences.
    Returns a list of adjusted p-values.
    
    Arguments:
    pos_seqs -- list, numpy array or pandas series of positive DNA sequences
    neg_seqs -- list, numpy array or pandas series of negative DNA sequences
    motifs -- list, numpy array or pandas series, a collection of motif patterns
        to be matched to seqs
    Keyword arguments:
    p_adjust -- method used to correct for multiple testing problem. Options are same as
        statsmodels.stats.multitest (default 'fdr_bh')
    alpha -- cutoff FDR/p-value to declare statistical significance (default 0.05)
    verbose -- verbosity argument (default False)
    allow_multi_match -- bool, whether to allow for counting multiple matchs (default False)
    Returns:
    pvals -- a list of p-values.
    """
    from scipy.stats import hypergeom
    import statsmodels.stats.multitest as multi
    
    
    pvals = []
    N = len(pos_seqs) + len(neg_seqs)
    K = len(pos_seqs)
    motif_count_all = count_motif_instances(pos_seqs+neg_seqs, motifs, allow_multi_match=allow_multi_match)
    motif_count_pos = count_motif_instances(pos_seqs, motifs, allow_multi_match=allow_multi_match)
    
    for motif in motifs:
        n = motif_count_all[motif]
        x = motif_count_pos[motif]
        pval = hypergeom.sf(x-1, N, K, n)
        if verbose:
            if pval < 1e-5:
                print("motif {}: N={}; K={}; n={}; x={}; p={}".format(motif, N, K, n, x, pval))
#         pvals[motif] = pval
        pvals.append(pval)
    
    # adjust p-value
    if p_adjust is not None:
        pvals = list(multi.multipletests(pvals,alpha=alpha,method=p_adjust)[1])
    return pvals

def filter_motifs(pos_seqs, neg_seqs, motifs, cutoff=0.05, return_idx=False, **kwargs):
    """
    Wrapper function for returning the actual motifs that passed the hypergeometric test.
    
    Arguments:
    pos_seqs -- list, numpy array or pandas series of positive DNA sequences
    neg_seqs -- list, numpy array or pandas series of negative DNA sequences
    motifs -- list, numpy array or pandas series, a collection of motif patterns
        to be matched to seqs
    Keyword arguments:
    cutoff -- cutoff FDR/p-value to declare statistical significance. (default 0.05)
    return_idx -- whether the indices of the motifs are only returned. (default False)
    **kwargs -- other input arguments
    
    Returns:
    list of filtered motifs (or indices of the motifs)
    """ 
    pvals = motifs_hypergeom_test(pos_seqs, neg_seqs, motifs, **kwargs)
    if return_idx:
        return [i for i, pval in enumerate(pvals) if pval < cutoff]
    else:
        return [motifs[i] for i, pval in enumerate(pvals) if pval < cutoff]


def sort_motif(motif_seqs):
    """
    @suthor Beyond
    Function:
        return keys, which sort in ascending order, 
                    and then in descending order in local region
    """
    keys = np.array(list(motif_seqs.keys()))
    motif_length = []
    motif_count = []

    for key in keys:
        length = len(key)
        count = len(motif_seqs[key]["seq_idx"])
        motif_length.append(length)
        motif_count.append(count)

    motif_length = np.array(motif_length)
    motif_count = np.array(motif_count)
    
    index = np.argsort(motif_length)

    keys = keys[index]
    motif_length = motif_length[index]
    motif_count = motif_count[index]
    
    motif_len_list = []
    for len_ in motif_length:
        if len not in motif_len_list:
            motif_len_list.append(len_)

    last = 0
    len_in_list = motif_length.shape[0]
    for len_ in motif_len_list:
        start_index = last
        for i in range(last, len_in_list):
            if i == len_in_list-1 or len_ != motif_length[i]:
                end_index = i
                last = end_index
                break
        
        # exclude the end_index when sorting
        motif_count_temp = motif_count[start_index : end_index]
        keys_temp = keys[start_index: end_index]

        # index_temp = np.argsort(motif_count_temp)
        index_temp = motif_count_temp.argsort()[::-1] # descend
        motif_count[start_index: end_index] = motif_count_temp[index_temp]
        keys[start_index: end_index] = keys_temp[index_temp]
    return keys

def merge_motifs(motif_seqs, min_len=5, align_all_ties=True, **kwargs):
    """
        Function to merge similar motifs in input motif_seqs.
        
        First sort keys of input motif_seqs based on length. For each query motif with length
        guaranteed to >= key motif, perform pairwise alignment between them.
        
        If can be aligned, find out best alignment among all combinations, then adjust start
        and end position of high attention region based on left/right offsets calculated by 
        alignment of the query and key motifs.
        
        If cannot be aligned with any existing key motifs, add to the new dict as new key motif.
        
        Returns a new dict containing merged motifs.
        
        Arguments:
        motif_seqs -- nested dict, with the following structure: 
            {motif: {seq_idx: idx, atten_region_pos: (start, end)}}
            where seq_idx indicates indices of pos_seqs containing a motif, and
            atten_region_pos indicates where the high attention region is located.
        
        Keyword arguments:
        min_len -- int, specified minimum length threshold for contiguous region 
            (default 5)
        
        align_all_ties -- bool, whether to keep all best alignments when ties encountered (default True)
        
        **kwargs -- other input arguments, may include:
            - cond: custom condition used to declare successful alignment.
                default is score > max of (min_len -1) and (1/2 times min length of two motifs aligned)
        
        Returns:
        merged_motif_seqs -- nested dict with same structure as `motif_seqs`
    """ 
    
    from Bio import Align
    
    ### TODO: modify algorithm to improve efficiency later
    aligner = Align.PairwiseAligner()
    aligner.internal_gap_score = -10000.0 # prohibit internal gaps
    
    merged_motif_seqs = {}

    # # using the motifs after sorting
    # keys = sort_motif(motif_seqs)
    # for motif in keys:
    # using the motif in random order
    for motif in sorted(motif_seqs, key=len): # query motif
        if not merged_motif_seqs: # if empty
            merged_motif_seqs[motif] = motif_seqs[motif] # add first one
        else: # not empty, then compare and see if can be merged
            # first create all alignment scores, to find out max
            alignments = []
            key_motifs = []
            for key_motif in merged_motif_seqs.keys(): # key motif
                if motif != key_motif: # do not attempt to align to self
                    # first is query, second is key within new dict
                    # first is guaranteed to be length >= second after sorting keys
                    alignment=aligner.align(motif, key_motif)[0] 
                    
                    # condition to declare successful alignment
                    cond = max((min_len -1), 0.5 * min(len(motif), len(key_motif))) 
                    
                    if 'cond' in kwargs:
                        cond = kwargs['cond'] # override
                        
                    if alignment.score >= cond: # exists key that can align
                        alignments.append(alignment)
                        key_motifs.append(key_motif)

            if alignments: # if aligned, find out alignment with maximum score and proceed
                best_score = max(alignments, key=lambda alignment: alignment.score)
                best_idx = [i for i, score in enumerate(alignments) if score == best_score]
                
                if align_all_ties:
                    for i in best_idx:
                        alignment = alignments[i]
                        key_motif = key_motifs[i]

                        # calculate offset to be added/subtracted from atten_region_pos
                        left_offset = alignment.aligned[0][0][0] - alignment.aligned[1][0][0] # always query - key
                        if (alignment.aligned[0][0][1] <= len(motif)) & \
                            (alignment.aligned[1][0][1] == len(key_motif)): # inside
                            right_offset = len(motif) - alignment.aligned[0][0][1]
                        elif (alignment.aligned[0][0][1] == len(motif)) & \
                            (alignment.aligned[1][0][1] < len(key_motif)): # left shift
                            right_offset = alignment.aligned[1][0][1] - len(key_motif)
                        elif (alignment.aligned[0][0][1] < len(motif)) & \
                            (alignment.aligned[1][0][1] == len(key_motif)): # right shift
                            right_offset = len(motif) - alignment.aligned[0][0][1]

                        # add seq_idx back to new merged dict
                        merged_motif_seqs[key_motif]['seq_idx'].extend(motif_seqs[motif]['seq_idx'])

                        # calculate new atten_region_pos after adding/subtracting offset 
                        new_atten_region_pos = [(pos[0]+left_offset, pos[1]-right_offset) \
                                                for pos in motif_seqs[motif]['atten_region_pos']]
                        merged_motif_seqs[key_motif]['atten_region_pos'].extend(new_atten_region_pos)

                else:
                    alignment = alignments[best_idx[0]]
                    key_motif = key_motifs[best_idx[0]]

                    # calculate offset to be added/subtracted from atten_region_pos
                    left_offset = alignment.aligned[0][0][0] - alignment.aligned[1][0][0] # always query - key
                    if (alignment.aligned[0][0][1] <= len(motif)) & \
                        (alignment.aligned[1][0][1] == len(key_motif)): # inside
                        right_offset = len(motif) - alignment.aligned[0][0][1]
                    elif (alignment.aligned[0][0][1] == len(motif)) & \
                        (alignment.aligned[1][0][1] < len(key_motif)): # left shift
                        right_offset = alignment.aligned[1][0][1] - len(key_motif)
                    elif (alignment.aligned[0][0][1] < len(motif)) & \
                        (alignment.aligned[1][0][1] == len(key_motif)): # right shift
                        right_offset = len(motif) - alignment.aligned[0][0][1]

                    # add seq_idx back to new merged dict
                    merged_motif_seqs[key_motif]['seq_idx'].extend(motif_seqs[motif]['seq_idx'])

                    # calculate new atten_region_pos after adding/subtracting offset 
                    new_atten_region_pos = [(pos[0]+left_offset, pos[1]-right_offset) \
                                            for pos in motif_seqs[motif]['atten_region_pos']]
                    merged_motif_seqs[key_motif]['atten_region_pos'].extend(new_atten_region_pos)

            else: # cannot align to anything, add to new dict as independent key
                merged_motif_seqs[motif] = motif_seqs[motif] # add new one
    
    return merged_motif_seqs

def make_window(motif_seqs, pos_seqs, window_size=24):
    """
    Function to extract fixed, equal length sequences centered at high-attention motif instance.
    
    Returns new dict containing seqs with fixed window_size.
    
    Arguments:
    motif_seqs -- nested dict, with the following structure: 
        {motif: {seq_idx: idx, atten_region_pos: (start, end)}}
        where seq_idx indicates indices of pos_seqs containing a motif, and
        atten_region_pos indicates where the high attention region is located.
    pos_seqs -- list, numpy array or pandas series of positive DNA sequences
    
    Keyword arguments:
    window_size -- int, specified window size to be final motif length
        (default 24)
    
    Returns:
    new_motif_seqs -- nested dict with same structure as `motif_seqs`s
    """ 

    new_motif_seqs = {}
    
    # extract fixed-length sequences based on window_size
    for motif, instances in motif_seqs.items():
        new_motif_seqs[motif] = {'seq_idx':[], 'atten_region_pos':[], 'seqs': []}
        for i, coord in enumerate(instances['atten_region_pos']):
            atten_len = coord[1] - coord[0]
            if (window_size - atten_len) % 2 == 0: # even
                offset = (window_size - atten_len) / 2
                new_coord = (int(coord[0] - offset), int(coord[1] + offset))
                if (new_coord[0] >=0) & (new_coord[1] < len(pos_seqs[instances['seq_idx'][i]])):
                    # append
                    new_motif_seqs[motif]['seq_idx'].append(instances['seq_idx'][i])
                    new_motif_seqs[motif]['atten_region_pos'].append((new_coord[0], new_coord[1]))
                    new_motif_seqs[motif]['seqs'].append(pos_seqs[instances['seq_idx'][i]][new_coord[0]:new_coord[1]])
            else: # odd
                offset1 = (window_size - atten_len) // 2
                offset2 = (window_size - atten_len) // 2 + 1
                new_coord = (int(coord[0] - offset1), int(coord[1] + offset2))
                if (new_coord[0] >=0) & (new_coord[1] < len(pos_seqs[instances['seq_idx'][i]])):
                    # append
                    new_motif_seqs[motif]['seq_idx'].append(instances['seq_idx'][i])
                    new_motif_seqs[motif]['atten_region_pos'].append((new_coord[0], new_coord[1]))
                    new_motif_seqs[motif]['seqs'].append(pos_seqs[instances['seq_idx'][i]][new_coord[0]:new_coord[1]])

    return new_motif_seqs


def get_Atten_Seq(attention_file, sequence_file):
    """
    Function:
        return the enhancer sequence and the non-enhancer sequence, respectively; and its their attention score 
        
    Key parameter:
        attention_file: the file of saved attention scores
        sequence_file:  the file of original sequence 
        And the relation between them is one to one, from top to down. 
    """
    
    df_raw = pd.read_csv(sequence_file, sep="\t",header=None, names=["text","label"]) 
    atten_scores = np.load(attention_file)

    sequences = np.array(list(df_raw["text"]))
    labels = np.array(list(df_raw["label"]))
    indexes = np.arange(labels.shape[0])
    
    neg_index = indexes[labels==0]
    pos_index = indexes[labels==1]
    
    neg_atten_scores = atten_scores[neg_index]
    pos_atten_scores = atten_scores[pos_index]
    
    neg_seq = sequences[neg_index]
    pos_seq = sequences[pos_index]
    
    return pos_seq, neg_seq, pos_atten_scores, neg_atten_scores


### make full pipeline
def motif_analysis(pos_seqs,
                   neg_seqs,
                   pos_atten_scores,
                   window_size = 24,
                   min_len = 4,
                   max_gap = 0,
                   pval_cutoff = 0.005,
                   min_n_motif = 11,
                   align_all_ties = True,
                   save_file_dir = None,
                   **kwargs
                  ):
 
    """
    Wrapper function of full motif analysis tool based on DNABERT-viz.
    
    Arguments:
    pos_seqs -- list, numpy array or pandas series of positive DNA sequences
    neg_seqs -- list, numpy array or pandas series of negative DNA sequences
    pos_atten_scores -- numpy array of attention scores for postive DNA sequence
    
    Keyword arguments:
    window_size -- int, specified window size to be final motif length
        (default 24)
    min_len -- int, specified minimum length threshold for contiguous region 
        (default 5)
    max_gap -- int, specified maximum gap between two contiguous region
        (default 5, which means no gap)
    pval_cutoff -- float, cutoff FDR/p-value to declare statistical significance. (default 0.005)
    min_n_motif -- int, minimum instance inside motif to be filtered (default 3)
    align_all_ties -- bool, whether to keep all best alignments when ties encountered (default True)
    save_file_dir -- str, path to save outputs (default None)
    **kwargs -- other input arguments, may include:
        - verbose: bool, verbosity controller
        - atten_cond: custom conditions to filter/select high attention 
            (list of boolean arrays)
        - return_idx: whether the indices of the motifs are only returned.
        - align_cond: custom condition used to declare successful alignment.
            default is score > max of (min_len -1) and (1/2 times min length of two motifs aligned)
    
    Returns:
    merged_motif_seqs -- nested dict, with the following structure: 
        {motif: {seq_idx: idx, atten_region_pos: (start, end)}}
        where seq_idx indicates indices of pos_seqs containing a motif, and
        atten_region_pos indicates where the high attention region is located.
    
    """ 
    from Bio import motifs
    from Bio.Seq import Seq
    
    verbose = False
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
    
    if verbose:
        print("*** Begin motif analysis ***")
    pos_seqs = list(pos_seqs)
    neg_seqs = list(neg_seqs)
    
    if verbose:
        print("* pos_seqs: {}; neg_seqs: {}".format(len(pos_seqs),len(neg_seqs)))
    
    assert len(pos_seqs) == len(pos_atten_scores)
    
    max_seq_len = len(max(pos_seqs, key=len))
    motif_seqs = {}
    
    ## find the motif regions
    if verbose:
        print("* Finding high attention motif regions")
    for i, score in enumerate(pos_atten_scores):
        seq_len = len(pos_seqs[i])
        score = score[0:seq_len]
        
        # handle kwargs
        if 'atten_cond' in kwargs:
            motif_regions = find_high_attention(score, min_len=min_len, max_gap=max_gap, cond=kwargs['atten_cond'])
        else:
            motif_regions = find_high_attention(score, min_len=min_len, max_gap=max_gap)
            
        for motif_idx in motif_regions:
            seq = pos_seqs[i][motif_idx[0]:motif_idx[1]]
            if seq not in motif_seqs:
                motif_seqs[seq] = {'seq_idx': [i], 'atten_region_pos':[(motif_idx[0],motif_idx[1])]}
            else:
                motif_seqs[seq]['seq_idx'].append(i)
                motif_seqs[seq]['atten_region_pos'].append((motif_idx[0],motif_idx[1]))

    print("before filtering, the length of motif:{}".format(len(motif_seqs)))
    # filter motifs
    return_idx = False
    if 'return_idx' in kwargs:
        return_idx = kwargs['return_idx']
        kwargs.pop('return_idx')
    
    if verbose:
        print("* Filtering motifs by hypergeometric test")
    motifs_to_keep = filter_motifs(pos_seqs, 
                                   neg_seqs, 
                                   list(motif_seqs.keys()), 
                                   cutoff = pval_cutoff, 
                                   return_idx=return_idx, 
                                   **kwargs)
    
    motif_seqs = {k: motif_seqs[k] for k in motifs_to_keep}
    
    print("After filtering, the length of motif:{}".format(len(motif_seqs)))


    # merge motifs
    if verbose:
        print("* Merging similar motif instances")
    if 'align_cond' in kwargs:
        merged_motif_seqs = merge_motifs(motif_seqs, min_len=min_len, 
                                         align_all_ties = align_all_ties,
                                         cond=kwargs['align_cond'])
    else:
        merged_motif_seqs = merge_motifs(motif_seqs, min_len=min_len,
                                         align_all_ties = align_all_ties)
    
    print("After meger, the number of group:{}".format(len(merged_motif_seqs)))
  
    # make fixed-length window sequences
    if verbose:
        print("* Making fixed_length window = {}".format(window_size))
    merged_motif_seqs = make_window(merged_motif_seqs, pos_seqs, window_size=window_size)
    
    # remove motifs with only few instances
    if verbose:
        print("* Removing motifs with less than {} instances".format(min_n_motif))
    merged_motif_seqs = {k: coords for k, coords in merged_motif_seqs.items() if len(coords['seq_idx']) >= min_n_motif}
    
    print("After number filter, the number of group:{}".format(len(merged_motif_seqs)))

    if save_file_dir is not None:
        if verbose:
            print("* Saving outputs to directory")
        os.makedirs(save_file_dir, exist_ok=True)
        for motif, instances in merged_motif_seqs.items():
            # saving to files
            with open(save_file_dir+'/motif_{}_{}.txt'.format(len(instances['seq_idx']), motif), 'w') as f:
                for seq in instances['seqs']:
                    f.write(seq+'\n')

            # make weblogo
            seqs = [Seq(v) for i,v in enumerate(instances['seqs'])]
            m = motifs.create(seqs)
            m.weblogo(save_file_dir+"/motif_{}_{}_weblogo.png".format(len(instances['seq_idx']), motif), format='png_print',
                             show_fineprint=False, show_ends=False, color_scheme='color_classic')
    
    return merged_motif_seqs
