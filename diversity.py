import sys
import time
from typing import Any, Callable, Iterable, List, Set, Union
import networkx as nx
import numpy as np
import utils
import os
sys.path.append("../img2vec_pytorch")
from img_to_vec import Img2Vec
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

=
def diversity(X: ElemList, idxs: List[int], dist: Callable[[Any, Any], float]) -> float:
    div_val = sys.float_info.max
    for id1 in idxs:
        for id2 in idxs:
            if id1 != id2:
                div_val = min(div_val, dist(X[id1], X[id2]))
    return div_val


ElemList = Union[List[utils.Element], List[utils.ElementSparse]]

#define type instance for use later in algorithm
class Instance:
    def __init__(self, k: int, mu: float, m: int):
        self.k = k
        self.mu = mu
        self.div = sys.float_info.max
        self.idxs = set()
        if m > 1:
            self.group_idxs = []
            for c in range(m):
                self.group_idxs.append(set())


#below is the code for algorithm1, for processing stream data points
def StreamDivMax(X: ElemList, k: int, dist: Callable[[Any, Any], float], eps: float, dmax: float, dmin: float) -> (List[int], float):
    Ins = []
    cur_d = dmax - 0.01
    while cur_d > dmin:
        ins = Instance(k, mu=cur_d, m=1)
        Ins.append(ins)
        cur_d *= (1.0 - eps)
    for x in X:
        for ins in Ins:
            if len(ins.idxs) == 0:
                ins.idxs.add(x.idx)
            elif len(ins.idxs) < ins.k:
                div_x = sys.float_info.max
                flag_x = True
                for y_idx in ins.idxs:
                    div_x = min(div_x, dist(x, X[y_idx]))
                    if div_x < ins.mu:
                        flag_x = False
                        break
                if flag_x:
                    ins.idxs.add(x.idx)
                    ins.div = min(ins.div, div_x)
    max_inst = None
    max_div = 0
    for ins in Ins:
        if len(ins.idxs) == k and ins.div > max_div:
            max_inst = ins
            max_div = ins.div
    return max_inst.idxs, max_inst.div 



#below is for algorithm 3, first convert the images into vectors, then use cosine simialrity to gather similairty measures
def getSimVect():
    input_path = './200H_images'

    print("Getting vectors for  images...\n")
    img2vec = Img2Vec()

    # For each image, store the filename and vector as key, value in a dictionary
    pics = {}
    for file in os.listdir(input_path):
        filename = os.fsdecode(file)
        img = Image.open(os.path.join(input_path, filename)).convert('RGB')
        vec = img2vec.get_vec(img)
        pics[filename] = vec

    available_filenames = ", ".join(pics.keys())
    pic_name = ""
    #calculate similarity
    sims[key] = cosine_similarity(pics[pic_name].reshape((1, -1)), pics[key].reshape((1, -1)))[0][0]

    d_view = [(v, k) for k, v in sims.items()]
    d_view.sort(reverse=True)
    for v, k in d_view:
        print(v, k)
        except Exception as e:
            print(e) 
