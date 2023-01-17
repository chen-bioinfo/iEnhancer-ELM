import re
import numpy as np
from motif_util import *

def calculate_pro(seq_dict):
    """
    Function:
        seq_dict: {index: {"A":0, "C":0, "G":0, "T":0} }

    Return:
        return matrix; 200*4
    """
    pro_list = []
    for key, value in seq_dict.items():
        count_A = value["A"]
        count_C = value["C"]
        count_G = value["G"]
        count_T = value["T"]

        nucleic_sum = count_A + count_C + count_T + count_G

        pro_A = count_A / nucleic_sum
        pro_C = count_C / nucleic_sum
        pro_G = count_G / nucleic_sum
        pro_T = count_T / nucleic_sum

        pro_list.append((pro_A, pro_C, pro_G, pro_T))
    return pro_list

def calculate(seqs):
    # ACGT
    """
    Function: 
        calculate the probability of different letters in each dimension
    
    Return:
        mat: 
            dict:{ i:{A:{}, C:{}, G:{}, T{}} }
    """
    seq_count = {}
    for i, seq in enumerate(seqs):
        for j, nucleic in enumerate(seq):
            if j not in seq_count:
                seq_count[j] = {"A":0, "C":0, "G":0, "T":0}
            nucleic = nucleic.upper()
            seq_count[j][nucleic] += 1

    seq_pro = calculate_pro(seq_count)
    return seq_pro

def save_TomTom_format(seq_pro, file_name, seq_name="ssc"):
    width = len(seq_pro)
    content = "MEME version 4" + "\n"
    content += "ALPHABET=ACGT" + "\n"
    content += "strands:+ -" + "\n"
    content += "MOTIF {}".format(seq_name) + "\n"
    content += "letter-probability matrix: alength=4 w={}".format(width) + "\n"
    for item in seq_pro:
        str_line = "{:.4f}".format(item[0]) + "  " + "{:.4f}".format(item[1]) + "  " + "{:.4f}".format(item[2]) + "  " + "{:.4f}".format(item[3]) + "\n"
        content += str_line
    file_write = open(file_name, "w")
    file_write.write(content)
    file_write.flush()
    file_write.close()

def get_sequence(enhancer_file, have_label=True):
    if have_label:
        df_raw = pd.read_csv(enhancer_file, sep="\t",header=None,names=["text","label"]) 
        enhancer_seq = np.array(list(df_raw["text"]))
        enhancer_label = np.array(list(df_raw["label"]))

        indexes = np.arange(enhancer_label.shape[0])

        pos_index = indexes[enhancer_label==1]
        neg_index = indexes[enhancer_label==0]
        
        pos_enhancer = enhancer_seq[pos_index]
        neg_enhancer = enhancer_seq[neg_index]

        return pos_enhancer, neg_enhancer
    else:
        df_raw = pd.read_csv(enhancer_file,header=None,names=["text"]) 
        enhancer_seq = np.array(list(df_raw["text"]))
        return enhancer_seq


def calculate_probability(filename):
    """
    function: 
        Key parameter:
            filename: it content  motif, comprised of lots of regions.
        
        Result: save to TomTom
    """
    files = os.listdir(filename)
    for item in files:
        result = re.match(".*\.txt", item)
        if result:
            file_name = filename + item
            enhancer_seq = get_sequence(file_name, have_label=False)
            enhancer_index_pro = calculate(enhancer_seq)

            strs = item.split(".")
            save_name = filename + strs[0] + "_TomTom.txt"
       
            # print(enhancer_seq)
            # print(save_name)
            save_TomTom_format(enhancer_index_pro, save_name, seq_name=strs[0].split("_")[2])
            

def all_write_in_one(filename, name):
    content = ""
    files = os.listdir(filename)
    index = 0
    for item in files:
        result = re.match(".*_TomTom\.txt", item)
        if result:
            file_name = filename+item
            file_write = open(file_name, "r")
            strs = file_write.readlines()
            if index == 0:
                for item_strs in strs:
                    content += item_strs
                index += 1
            else:
                for j, item_strs in enumerate(strs):
                    if j >=3:
                        content += item_strs
            content += "\n"
    
    write_filename = filename + name + "all_TomTom.txt"
    write_file = open(write_filename, "w")
    write_file.write(content)
    write_file.flush()
    write_file.close()