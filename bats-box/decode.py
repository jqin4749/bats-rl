import os, sys

from numpy.core.fromnumeric import reshape
sys.path.insert(1,os.getcwd())
import utils as ut
import numpy as np

def rank(a):
    return np.linalg.matrix_rank(a)
def full_rank(a):
    if rank(a) == min(a.shape):
        return True
    else:
        return False

class decoder:
    def __init__(self) -> None:
        self.num_decoded_pkts_ = 0
        self.total_num_vnodes_ = 0
        self.inactivated_ = False
        self.inact_vns_ = []
        self.inact_y_ = [] # n , pk_size
        self.inact_con_ = [] # n , vns (max inact_num)
    def reset(self):
        self.num_decoded_pkts_ = 0
        self.total_num_vnodes_ = 0
        self.inactivated_ = False
        self.inact_vns_ = []
        self.inact_y_ = [] 
        self.inact_con_ = []

    def stop_condition(self,graph:ut.tanner_graph):
        if self.last_num_decoded_pkts_ == self.num_decoded_pkts_:
            return False
        if self.total_num_vnodes_ == self.num_decoded_pkts_:
            return False
        
        return True


    def decode(self,graph:ut.tanner_graph,inactivation=False,silent=False):
        assert graph.cnode_loaded_ == True,'[decoder] checknodes are not loaded'
        self.bsize_ = graph.bsize_
        self.n_vns_ = graph.n_vns_
        self.n_cns_ = graph.n_cns_
        self.total_num_vnodes_ = len(graph.vnodes_)
        self.last_num_decoded_pkts_ = -1
        self.silent_ = silent
        while self.stop_condition(graph):
            self.last_num_decoded_pkts_ = self.num_decoded_pkts_
            for b in graph.cnodes_:
                if b.all_lost_ == True or len(b.edges_) == 0:
                    continue
                self.do_decode(b,silent)
        if silent == False:
            print('[decode]BP decoding stops. %d/%d packets decoded'%(self.num_decoded_pkts_,self.total_num_vnodes_))
            print('decoded batch:',[i.id_ for i in graph.cnodes_ if i.decoded_])
            print('undecoded batch:',[i.id_ for i in graph.cnodes_ if i.decoded_ == False])
        if inactivation and self.num_decoded_pkts_ != self.total_num_vnodes_:
            if silent == False:
                print('[decode] Inactivation decoding now starts')
            self.inactivated_ = True
            self.inactivation_decode(graph,silent)
            if silent == False:
                print('[decode]inact decoding stops. %d/%d packets decoded'%(self.num_decoded_pkts_,self.total_num_vnodes_))

    def find_keep_idx(self, h):
        idx_list = []
        span = []
        last_rk = 0
        for idx, i in enumerate(h.T):
            span_cat = span + [i]
            new_rk = rank(np.concatenate([np.expand_dims(i,axis=0) for i in span_cat],axis=0))
            if new_rk == last_rk + 1:
                last_rk = new_rk
                span.append(i)
                idx_list.append(idx)
        return idx_list

    def do_decode(self, cn:ut.cnode,silent=False):
        if len(cn.get_undecoded_edges()) == 0:
            cn.decoded_ = True
            return
        if cn.decoded_ == True:
            return
        
        num_undecoded_edges = len(cn.get_undecoded_edges())
        h = cn.get_h()
        keep_idx = self.find_keep_idx(h)
        h = h[:,keep_idx]
        batch = cn.get()[keep_idx,:].T 
        g = np.concatenate([np.expand_dims(e.g_,axis=0) for e in cn.get_undecoded_edges()],axis=0)
        gh = np.matmul(g,h[:,:num_undecoded_edges]) 
        batch_rk = rank(gh)
        if batch_rk < num_undecoded_edges:
            return # cannot be decoded
        if silent == False:
            print('batch%d with rank %d is decodable'%(cn.id_,batch_rk))
        
        assert full_rank(gh), '[decoder] gh is not full rank. gh:shape:(%d,%d). rk: %d\ng:(%d,%d). rk: %d\nh:(%d,%d). rk: %d'%(
                                                                                                                gh.shape[0],gh.shape[1],rank(gh),
                                                                                                                g.shape[0],g.shape[1],rank(g),
                                                                                                                h.shape[0],h.shape[1],rank(h))
        gh_inv = np.linalg.inv(gh)
        pkt_decoded = np.matmul(batch[:,:num_undecoded_edges],gh_inv)
        for idx, e in enumerate(cn.get_undecoded_edges()):
            e.vn_.load(pkt_decoded[:,idx]) 
            e.vn_.decoded_ = True
            e.decoded_ = True
            for ee in e.vn_.edges_:
                if ee.cn_.id_ == cn.id_:
                    continue
                ee.decoded_ = True
                self.backprop(ee)
        self.num_decoded_pkts_ += num_undecoded_edges
        cn.decoded_ = True
    
    def backprop(self,eg):
        g = np.expand_dims(eg.g_,axis=0)
        H = eg.cn_.get_h()
        pk = np.expand_dims(eg.vn_.get(),axis=-1)
        assert pk.ndim == 2, '[backprop] dimension error'
        gH = np.matmul(g,H)
        pgH = np.matmul(pk,gH)
        eg.cn_.load(np.subtract(eg.cn_.get().T,pgH).T)
        if self.inactivated_:
            v_coef_gH = np.matmul(np.expand_dims(eg.vn_.inact_coef_,axis=-1),gH)
            if len(eg.cn_.inact_coef_) == 0:
                eg.cn_.inact_coef_ = ut.gf(np.zeros((self.bsize_, self.n_vns_),dtype=int))
            eg.cn_.inact_coef_ = np.subtract(eg.cn_.inact_coef_, v_coef_gH.T)
        
    def inactivation_decode(self,graph:ut.tanner_graph,silent=False):
        while 1:
            vn = self.find_vn_to_inactivate(graph)
            if vn == None: 
                self.decode_inact_symbols(graph)
                break
            vn.inactivated_ = True
            self.inact_vns_.append(vn)
            self.prop_inactivation_to_cn(graph,vn)
            self.do_decode_inactivation(graph,silent)
            if  len(self.get_unprocessed_vns(graph)) == 0:
                self.decode_inact_symbols(graph,about_to_exit=True)
                break
            if len(self.inact_vns_) == len(self.inact_con_):
                self.decode_inact_symbols(graph)
        return

    # def find_vn_to_inactivate(self, graph:ut.tanner_graph):
    #     # find vnode with the most edges
    #     vns = [v for v in graph.vnodes_ if not v.decoded_ and not v.partial_decoded_ and not v.inactivated_]
    #     if len(vns)==0: return None
    #     max_eg = 0
    #     max_v = vns[0]
    #     for v in vns:
    #         if len(v.edges_) > max_eg:
    #             max_eg = len(v.edges_)
    #             max_v = v
    #     return max_v
    
    def find_vn_to_inactivate(self, graph:ut.tanner_graph):
        # find the cnode with smallest rank defficiency
        rd = 1000
        best_cn = graph.cnodes_[0]
        for c in graph.cnodes_:
            if c.decoded_:
                continue
            rd_new = self.cal_rank_defficiency(c)
            if rd_new < rd and rd_new != 0:
                best_cn = c
                rd = rd_new
        vns = [e.vn_ for e in best_cn.edges_ if not e.vn_.decoded_ and not e.vn_.partial_decoded_ and not e.vn_.inactivated_]
        if len(vns)==0: return None
        max_eg = 0
        max_v = vns[0]
        for v in vns:
            if len(v.edges_) > max_eg:
                max_eg = len(v.edges_)
                max_v = v        
        return max_v

    def cal_rank_defficiency(self,cn:ut.cnode):
        h = cn.get_h()
        h_rk = rank(h)
        undecoded_edges = cn.get_undecoded_edges()
        return len(undecoded_edges) - h_rk

    def prop_inactivation_to_cn(self,graph:ut.tanner_graph, vn:ut.vnode):
        # update edges. 
        if self.silent_ == False:
            print('[%d] adding vn%d'%(len(self.inact_vns_),vn.id_))
        for e in vn.edges_:
            e.decoded_ = True
            cn = e.cn_
            h = cn.get_h()
            gh = np.matmul(np.expand_dims(e.g_,axis=0),h) # (1, h_rk)
            if len(cn.inact_coef_) == 0:
                cn.inact_coef_ = ut.gf(np.zeros((graph.bsize_, graph.n_vns_),dtype=int))
            gh = gh.reshape(-1)
            cn.inact_coef_[:,len(self.inact_vns_)-1] = np.subtract(cn.inact_coef_[:,len(self.inact_vns_)-1], gh)
           
        return
    
    def get_unprocessed_vns(self, graph:ut.tanner_graph):
        # total vns - decoded vns - inactivated vns - partially decoded vns
        return [v for v in graph.vnodes_ if not v.decoded_ and not v.partial_decoded_ and not v.inactivated_]

    def decode_inact_symbols(self,graph:ut.tanner_graph,about_to_exit=False):
        if len(self.inact_vns_) > len(self.inact_con_) or len(self.inact_vns_) == 0:
            if self.silent_ == False:
                print('inact decode fail. inact_vns_ %d > inact_con_ %d'%(len(self.inact_vns_),len(self.inact_con_)))
            return # cannot decode
        if self.silent_ == False:
            print('inact_vns_ %d > inact_con_ %d'%(len(self.inact_vns_),len(self.inact_con_)))
        
        y = np.concatenate([np.expand_dims(i,axis=0) for i in self.inact_y_],axis=0)[:len(self.inact_vns_)]
        con = np.concatenate([np.expand_dims(i[:len(self.inact_vns_)],axis=0) for i in self.inact_con_],axis=0)[:len(self.inact_vns_)].T
        if not full_rank(con): return  
        con_inv = np.linalg.inv(con)
        inact_pkt = np.matmul(y.T,con_inv)
       
        for idx, v in enumerate(self.inact_vns_):
            v.load(inact_pkt[:,idx])
            v.decoded_ = True
            v.inactivated_ = False
            self.num_decoded_pkts_ += 1

        for v in graph.vnodes_:
            if not v.partial_decoded_:
                continue
            pkt = np.matmul(inact_pkt, np.expand_dims(v.inact_coef_[:inact_pkt.shape[-1]],axis=-1)).reshape(-1)
            pkt = np.subtract(v.get().reshape(-1), pkt)
            v.load(pkt)
            v.decoded_ = True
            v.partial_decoded_ = False
            self.num_decoded_pkts_ += 1
        self.inact_con_ = []
        self.inact_vns_ = []
        self.inact_y_ = []
        return


    def do_decode_inactivation(self,graph:ut.tanner_graph,silent=False):
        current_num_patrial_decoded_vn = len([v for v in graph.vnodes_ if v.partial_decoded_])
        last_num_patrial_decoded_vn = -1
        while current_num_patrial_decoded_vn != last_num_patrial_decoded_vn:
            for cn in graph.cnodes_:
                if cn.all_lost_ == True or len(cn.edges_) == 0:
                    continue
                self.decode_cn_inaction(cn,silent)
            last_num_patrial_decoded_vn = current_num_patrial_decoded_vn
            current_num_patrial_decoded_vn = len([v for v in graph.vnodes_ if v.partial_decoded_])

    def decode_cn_inaction(self, cn:ut.cnode,silent):
        if cn.decoded_ == True:
            return
        undecoded_edges = cn.get_undecoded_edges()
        if len(undecoded_edges) == 0:
            cn.decoded_ = True
            self.add_cn_as_constraint(cn)
            return
          
        num_undecoded_edges = len(undecoded_edges)
        h = cn.get_h()
        # keep_idx = self.find_keep_idx(h)
        # h = h[:,keep_idx]
        batch = cn.get().T 
        g = np.concatenate([np.expand_dims(e.g_,axis=0) for e in undecoded_edges],axis=0)
        gh = np.matmul(g,h[:,:num_undecoded_edges]) 
        batch_rk = rank(gh)
        h_rk = rank(h)
        if batch_rk < num_undecoded_edges:
            # print('batch%d cannot be decoded. %d < %d'%(cn.id_,batch_rk,num_undecoded_edges))
            return # cannot be decoded
        if silent == False:
            print('[inact] batch%d with rank %d is decodable'%(cn.id_,batch_rk))
    
        assert full_rank(gh), '[decoder] gh is not full rank. gh:shape:(%d,%d). rk: %d\ng:(%d,%d). rk: %d\nh:(%d,%d). rk: %d'%(
                                                                                                                gh.shape[0],gh.shape[1],rank(gh),
                                                                                                                g.shape[0],g.shape[1],rank(g),
                                                                                                                h.shape[0],h.shape[1],rank(h))
        gh_inv = np.linalg.inv(gh)
        pkt_decoded = np.matmul(batch[:,:num_undecoded_edges],gh_inv)
        
        inact_coef = np.matmul(cn.inact_coef_[:num_undecoded_edges,:].T, 
                                gh_inv)
     
        for idx, e in enumerate(undecoded_edges):
            e.vn_.load(pkt_decoded[:,idx])
            e.vn_.partial_decoded_ = True
            e.vn_.inact_coef_ = inact_coef[:,idx]
            e.decoded_ = True
            for ee in e.vn_.edges_:
                if ee.cn_.id_ == cn.id_:
                    continue
                ee.decoded_ = True
                self.backprop(ee)
        
        cn.decoded_ = True
        # add linear constraints
        num_cons = h_rk - num_undecoded_edges
        if  num_cons > 0:
            y = batch[:,num_undecoded_edges:h_rk] 
            con = cn.inact_coef_[num_undecoded_edges:h_rk,:] 
            h_ = h[:,num_undecoded_edges:h_rk]
            gh_ = np.matmul(g,h_)
            y = np.subtract(y, np.matmul(pkt_decoded,gh_))
            con = np.subtract(con.T, np.matmul(inact_coef,gh_)).T

            self.add_constraints(y,con)

        return     

    def add_cn_as_constraint(self,cn:ut.cnode):
        h = cn.get_h()
        h_rk = rank(h)
        batch = cn.get()[:h_rk].T 
        y = batch 
        con = cn.inact_coef_[:h_rk,:]
        self.add_constraints(y,con)
        return  
    
    def add_constraint(self, y, con):
        # Check rank
        rk = rank(np.concatenate([np.expand_dims(i,axis=0) for i in self.inact_con_],axis=0)) if len(self.inact_con_) != 0 else 0
        con_cat = self.inact_con_ + [con]
        rk2 = rank(np.concatenate([np.expand_dims(i,axis=0) for i in con_cat],axis=0))
        if rk + 1 == rk2:
            self.inact_con_.append(con)
            self.inact_y_.append(y)
            # print(con.shape,y.shape)
        return
    def add_constraints(self, y, con):
        if con.ndim == 1:
            con = np.expand_dims(con,axis=0)
        if y.ndim == 1:
            y = np.expand_dims(y,axis=-1)
        # print(con.shape,y.shape)
        
        for i,j in zip(y.T,con):
            self.add_constraint(i, j)

    