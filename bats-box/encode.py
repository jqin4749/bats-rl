import os, sys
sys.path.insert(1,os.getcwd())
import utils as ut
import numpy as np

class encoder():
    def __init__(self) -> None:
        pass
    def encode(self,graph:ut.tanner_graph):
        assert graph.vnode_loaded_ == True,'[encoder] varibale nodes are not loaded'
        for b in graph.cnodes_:
            if len(b.edges_) == 0:
                continue
            pkts = np.concatenate([np.expand_dims(i,axis=0) for i in b.get_v()],axis=0).T
            G = b.get_G()
            b.load(np.matmul(pkts, G).T)
        graph.cnode_loaded_ = True