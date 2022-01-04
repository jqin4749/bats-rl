from matplotlib.colors import rgb2hex
import numpy as np
import os,sys
# sys.path.insert(1,os.getcwd())
import matplotlib.pyplot as plt
from galois import GF

gf256 = GF(2**8)
def gf(a):
    return gf256(a)

class vnode:
    def __init__(self,label='vnode', pk = [],pk_size=1024):
        pk_ = np.array(pk)
        assert pk_.ndim == 1, '[vnode]:pk dimension error:%d'%(pk_.dim)  
        self.pk_ = pk_
        self.label_ = label
        self.decoded_ = False
        self.edges_ = []
        self.id_ = int(label.split('v')[1])
        self.pk_size_ = pk_size
        self.inactivated_ = False
        self.partial_decoded_ = False
        self.unknown_vn_involved_ = []
        self.inact_coef_ = []

    def shape(self):
        return self.pk_.shape
    def len(self):
        return len(self.pk_)
    def load(self,pk):
        pk_ = np.array(pk)
        assert pk_.ndim == 1, '[vnode]:pk dimension error:%d'%(pk_.dim)  
        self.pk_ = gf(pk_)
    def get(self):
        return self.pk_
    def add_edge(self,edge):
        self.edges_.append(edge)
    def pksize(self):
        return self.pk_size_

class cnode:
    def __init__(self,label='cnode',bsize = 16, batch = []) -> None:
        batch_ = np.array(batch)
        self.batch_ = batch_
        self.label_ = label
        self.decoded_ = False
        self.edges_ = []
        self.id_ = int(label.split('c')[1])
        self.h_ = []
        self.bsize_ = bsize
        self.all_lost_ = False
        self.h_ = gf(np.identity(self.bsize(),dtype=int))
        self.inact_coef_ = []
        
    def bsize(self):
        return self.bsize_
    def pksize(self):
        return self.batch_.shape[1] # column major
    def len(self):
        return self.len(self.batch_)
    def shape(self):
        return self.batch_.shape
    def load(self,batch):
        batch_ = np.array(batch)
        assert batch_.ndim == 2, '[cnode]:batch dimension error:%d'%(batch_.ndim)
        assert batch.shape[0] == self.bsize_, '[cnode] batch need to be column major'
        self.batch_ = gf(batch_)
        
    def get(self):
        return self.batch_
    def add_edge(self,edge):
        self.edges_.append(edge)
    def get_G(self):
        return np.concatenate([np.expand_dims(i.g_,axis=0) for i in self.edges_],axis=0)
    def get_v(self):
        return [e.vn_.get() for e in self.edges_]
    def get_h(self):
        return self.h_
    def load_h(self,h):
        self.h_ = h
    def get_decoded_edges(self):
        return [e for e in self.edges_ if e.decoded_ == True]
    def get_undecoded_edges(self):
        return [e for e in self.edges_ if e.decoded_ == False]
    def get_unknow_vns(self):
        unknow_vns = []
        for e in self.edges_:
            if e.vn_.inactivated_:
                unknow_vns.append(e.vn_.id_)
            if e.vn_.patrial_decoded_:
                unknow_vns += e.vn_.unknown_vn_involved_
        unknow_vns = remove_repeat(unknow_vns)
        return unknow_vns

def remove_repeat(a):
    d = {}
    for i in a:
        d[i] = 0
    return list(d.keys())

class edge:
    def __init__(self,vn, cn,bsize=16,label='edge') -> None:
        self.label_ = label
        self.vn_ = vn
        self.cn_ = cn
        self.decoded_ = vn.decoded_
        self.bsize_ = bsize
        # generate g
        self.gen_g()
        vn.add_edge(self)
        cn.add_edge(self)
    def gen_g(self):
        self.g_ = gf(np.random.randint(256,size=self.bsize_))
        

import networkx as nx

def draw_tanner(n1,n2,egs):
    sanity_check_for_raw_graph(n1,n2,egs)
    B = nx.Graph()
    B.add_nodes_from(n1, bipartite=0) # variable nodes
    B.add_nodes_from(n2, bipartite=1) # check nodes
    B.add_edges_from(egs)
    fig = plt.figure(1,figsize=(10, 10))
    pos = {}
    pos.update( (n, (1, i)) for i, n in enumerate(n1) ) 
    pos.update( (n, (2, i)) for i, n in enumerate(n2) ) 
    nx.draw_networkx_nodes(B,nodelist = n1, label = n1, pos = pos)
    nx.draw_networkx_nodes(B,nodelist = n2, label = n2, pos = pos,node_shape='s')
    nx.draw_networkx_edges(B, pos = pos)
    nx.draw_networkx_labels(B, pos = pos)
    return fig

def sanity_check_for_raw_graph(n1,n2,egs):
    for i,j in egs:
        assert i in n1, '[sanity_check_for_raw_graph] vnode not exist %s'%i
        assert j in n2, '[sanity_check_for_raw_graph] cnode not exist %s'%j

class tanner_graph:
    def __init__(self, n_vns, n_cns, raw_con, bsize=16,pk_size=1024, label = 'tanner'):
        self.n_vns_ = n_vns 
        self.n_cns_ = n_cns 
        self.raw_con_ = raw_con 
        self.vnodes_ = [] # [vnode,..]
        self.cnodes_ = [] # [cnode,..]
        self.edges_ = []
        self.vnode_loaded_ = False
        self.cnode_loaded_ = False
        self.label_ = label
        self.bsize_ = bsize
        self.pk_size_ = pk_size
        self.build_graph()
        return
    def __str__(self): 
        str_to_return = 'Total packets:%d\nTotal batches:%d\n\n'%(self.n_vns_,self.n_cns_)
        str_to_return += 'Batch view:\n'
        for i in self.cnodes_:
            str_to_return += 'batch %d: connect to %d variable node\n\t-->'%(i.id_,len(i.edges_))
            for j in i.edges_:
                str_to_return += '%s, '%j.vn_.label_
            str_to_return += '\n'
        str_to_return += '\n\nVariable node view:\n'
        for i in self.vnodes_:
            str_to_return += 'vnode %d: connect to %d check node\n\t-->'%(i.id_,len(i.edges_))
            for j in i.edges_:
                str_to_return += '%s, '%j.cn_.label_
            str_to_return += '\n'
        str_to_return += '\n'
        return str_to_return
    def info(self,long = False):
        if long:
            print(self.__str__())
            return
        print('Graph Name:%s\nTotal packets:%d\nTotal batches:%d'%(self.label_,self.n_vns_,self.n_cns_))
        print('Total number of edges:%d'%len(self.edges_))
        if self.vnode_loaded_:
            print('Packet size:%d'%self.get_pkt_size())
        else:
            print('Variable node not loaded')
        if self.cnode_loaded_:
            print('Batch size:%d'%self.get_batch_size())
        else:
            print('Checknode not loaded')
        unconnected_vnode = [i for i in self.vnodes_ if len(i.edges_)==0]
        print('Unconnected vnodes:%d/%d'%(len(unconnected_vnode),self.n_vns_))
        if len(unconnected_vnode) != 0 :
            print('They are:',unconnected_vnode) 


        
    def draw(self):
        draw_tanner(self.get_vns(),self.get_cns(),self.get_connection())

    def get_pkt_size(self):
        return self.vnodes_[0].len()
    def get_batch_size(self):
        return self.bsize_
    def get_real_cv_size(self):
        c = {}
        v = {}
        for i,j in self.raw_con_:
            v[i] = 1
            c[j] = 1
        return len(c.keys()), len(v.keys())

    def build_graph(self):
        # self.n_cns_, self.n_vns_ = self.get_real_cv_size()
        # create cnode, vnode objects
        for i in range(self.n_vns_):
            self.vnodes_.append(vnode(label='v%d'%i,pk_size=self.pk_size_))
        for i in range(self.n_cns_):
            self.cnodes_.append(cnode(label='c%d'%i,bsize=self.bsize_))
        for i,j in self.raw_con_:
            self.edges_.append(edge(self.vnodes_[i], self.cnodes_[j],self.bsize_,label='v%d--c%d'%(i,j),)) 
        # make sure all G are full rank
        for i in range(self.n_cns_):
            if len(self.cnodes_[i].edges_) == 0:
                continue
            rg_got = np.linalg.matrix_rank(self.cnodes_[i].get_G())
            rg_expect = min(self.bsize_, len(self.cnodes_[i].edges_))
            while rg_got != rg_expect:
                # print('[build_graph] batch %d: regenerating G (rank=%d/%d)'%(self.cnodes_[i].id_,rg_got,rg_expect))
                for g in self.cnodes_[i].edges_:
                    g.gen_g()
                rg_got = np.linalg.matrix_rank(self.cnodes_[i].get_G())

        sanity_check_for_raw_graph(self.get_vns(),self.get_cns(),self.get_connection())
        
    def get_vns(self):
        return [i.label_ for i in self.vnodes_]
    def get_cns(self):
        return [i.label_ for i in self.cnodes_]
    def get_connection(self):
        return [(eg.vn_.label_, eg.cn_.label_) for eg in self.edges_] # [(vnode,cnode),...]
    
    def load_vnodes(self,vnodes):
        self.vnode_loaded_ = True
        for idx, i in enumerate(vnodes):
            self.vnodes_[idx].load(i)
    def load_cnodes(self,cnodes):
        self.cnode_loaded_ = True
        for idx, i in enumerate(cnodes):
            self.cnodes_[idx].load(i)
    def get_vnodes(self):
        return [vn.get() for vn in self.vnodes_]
    
    def get_cnodes(self):
        return [cn.get() for cn in self.cnodes_]

    def copy(self, with_batch = False): 
        t = tanner_graph(self.n_vns_,self.n_cns_,self.raw_con_,self.bsize_,self.pk_size_,'%s_copy'%self.label_) # create an identical graph without data
        for e_self, e_ in zip(self.edges_,t.edges_):
            e_.g_ = e_self.g_
        if with_batch:
            t.load_batches(self)
        return t

    def load_batches(self,graph):
        batch = graph.get_cnodes()
        self.load_cnodes(batch)
        for b_self,b_ in zip(self.cnodes_, graph.cnodes_):
            b_self.h_ = b_.h_
    

def gen_degree(n_pkt=64,dist_name='./deg_dist2.txt'):
    pdf = np.zeros(n_pkt)
    with open(dist_name) as f:
        line = f.readlines()
    for i in line:
        idx = int(i.split(' ')[0])
        pdf[idx] = float(i.split(' ')[1])
    pdf = pdf/np.sum(pdf)
    d = np.random.choice(np.arange(1,len(pdf)+1),1,p=pdf)[0]
    return int(np.ceil(n_pkt*d/len(pdf)))

def sample_pkts(dgs,n_pkt=64):
    sample_set = [np.random.choice(np.arange(0,n_pkt),dg,replace=False) for dg in dgs]
    return sample_set

def gen_tanner(n_pkt,n_batch,bsize,pk_size,silent=True,label='tanner',deg_dist='preset',dgs_=None):
    if deg_dist == 'preset':
        dgs = [gen_degree(n_pkt) for _ in range(n_batch)]
    elif deg_dist == 'uniform':
        dgs = list(np.random.choice(np.arange(1,n_pkt+1),n_batch,replace=False))
    elif deg_dist == 'hardcoded':
        dgs = dgs_
    else:
        dgs = [gen_degree(n_pkt,deg_dist) for _ in range(n_batch)]

    sample_set = sample_pkts(dgs,n_pkt)
    raw_con = [(v,c) for c in range(n_batch) for v in sample_set[c]]
    if silent == False:
        print('degs:',dgs)
        # print('sample_set:',sample_set)
    return tanner_graph(n_pkt,n_batch,raw_con,bsize,pk_size,label)

def gen_tanner_all_connected(n_pkt,n_batch,bsize,pk_size,silent=True,label='tanner',deg_dist='preset',dgs_=None):
    while 1:
        t = gen_tanner(n_pkt,n_batch,bsize,pk_size,silent,label,deg_dist,dgs_)
        ophant_vnode = [i for i in t.vnodes_ if len(i.edges_) == 0]
        if len(ophant_vnode) == 0:
            return t


def gen_tanner_from_binary_graph(graph_bin_, pk_size, bsize):
    assert graph_bin_.ndim == 2, '[gen_tanner_from_binary_graph] dimension error'
    graph_bin = np.concatenate([np.expand_dims(i,axis=0) for i in graph_bin_ if np.sum(i)!=0],axis=0)
    graph_bin = np.concatenate([np.expand_dims(i,axis=0) for i in graph_bin.T if np.sum(i)!=0],axis=0).T
    n_cns = graph_bin.shape[0]
    n_vns = graph_bin.shape[1]
    raw_con = []
    for row, i in enumerate(graph_bin):
        for col,j in enumerate(i):
            if j == 1:
                raw_con.append((col,row))
    return tanner_graph(n_vns,n_cns,raw_con,bsize,pk_size)

def gen_binary_graph_from_con(n_cns, n_vns, con):
    bg = np.zeros((n_cns,n_vns),dtype=int)
    for v, c in con:
        bg[c,v] = 1
    return bg

def gen_binary_graph_from_dist(n_pkt,n_batch,deg_dist='preset',dgs_=None):
    if deg_dist == 'preset':
        dgs = [gen_degree(n_pkt) for _ in range(n_batch)]
    elif deg_dist == 'uniform':
        dgs = list(np.random.choice(np.arange(1,n_pkt+1),n_batch,replace=False))
    elif deg_dist == 'hardcoded':
        dgs = dgs_
    else:
        dgs = [gen_degree(n_pkt,deg_dist) for _ in range(n_batch)]

    sample_set = sample_pkts(dgs,n_pkt)
    raw_con = [(v,c) for c in range(n_batch) for v in sample_set[c]]
    return gen_binary_graph_from_con(n_batch, n_pkt, raw_con)

def gen_binary_graph_from_dist_all_connected(n_pkt,n_batch,deg_dist='preset',dgs_=None):
    while 1:
        g = gen_binary_graph_from_dist(n_pkt,n_batch,deg_dist,dgs_)
        if len([i for i in g.T if np.sum(i) == 0]) == 0:
            return g
    