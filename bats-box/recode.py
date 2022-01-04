import os, sys
sys.path.insert(1,os.getcwd())
import utils as ut
import numpy as np

def drop(bsize,loss_rate):
    idx = []
    for i in range(bsize):
        if np.random.rand() < loss_rate:
            idx.append(i)
    return idx

class recoder:
    def __init__(self) -> None:
        pass
    def recode(self,graph:ut.tanner_graph, num_hops=1, loss_rate=0,silent=True):
        assert graph.cnode_loaded_ == True,'[recoder] checknodes are not loaded'
        for hop in range(num_hops):
            h = ut.gf(np.random.randint(256,size=(graph.bsize_,graph.bsize_)))
            while np.linalg.matrix_rank(h) != graph.bsize_:
                # if silent == False:
                #     print('[recode]regenerate h')
                h = ut.gf(np.random.randint(256,size=(graph.bsize_,graph.bsize_)))
            for b in graph.cnodes_:
                if b.all_lost_ == True or len(b.edges_) == 0:
                    continue
                drop_idx = drop(graph.bsize_,loss_rate)
                # if silent == False:
                #     print('[recode] batch%d drop %d pkts'%(b.id_,len(drop_idx)),drop_idx)
                keep_index = [i for i in range(graph.bsize_) if i not in drop_idx]
                if len(keep_index) == 0:
                    b.all_lost_ = True
                    continue
                batch = b.get()[keep_index].T 
                b.load(np.matmul(batch, h[keep_index]).T) # data is column major
                b.load_h(np.matmul(b.get_h()[:,keep_index],h[keep_index])) # coef vector is not column major
        
