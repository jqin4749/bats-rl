import os, sys
sys.path.insert(1,os.getcwd())
import utils as ut
import numpy as np
from encode import encoder
from recode import recoder
from decode import decoder

class bats_box:
    def __init__(self,graph:ut.tanner_graph = None) -> None:
        self.data_ = []
        self.dout_ = []
        self.graph_ = graph
        self.encoder_ = encoder()
        self.recoder_ = recoder()
        self.decoder_ = decoder()
        self.num_decoded_pkts_ = 0
        self.pk_num_ = 0
        self.pk_size_ = 0
        self.batch_num_ = 0
        self.bsize_ = 0
        self.edge_num_ = 0

    def update_graph(self,graph:ut.tanner_graph):
        self.graph_ = graph
        self.pk_num_ = len(graph.vnodes_)
        self.pk_size_ = graph.vnodes_[0].pksize()
        self.batch_num_ = len(graph.cnodes_)
        self.bsize_ = graph.cnodes_[0].bsize()
        self.edge_num_ = len(graph.edges_)
        # get info: num_vnode, num_cnode, pk_size, batch_size, num_edges

    def get_decoded_pkt_idx(self,graph:ut.tanner_graph):
        return [idx for idx, vn in enumerate(graph.vnodes_) if vn.decoded_ == True]

    def run(self,num_hops=1,loss_rate=0.1,inactivation=False,silent=False):
        assert self.graph_ != None, '[bats_box] graph not loaded'
        self.decoder_.reset()
        # generate data
        self.data_ = np.random.randint(256,size=(self.pk_size_,self.pk_num_))
        self.graph_.load_vnodes(self.data_.T)
        self.encoder_.encode(self.graph_)
        graph_to_decode = self.graph_.copy(with_batch=True)
        self.recoder_.recode(graph_to_decode,num_hops,loss_rate,silent)
        self.decoder_.decode(graph_to_decode,inactivation,silent)
        self.dout_ = graph_to_decode.get_vnodes()
        self.num_decoded_pkts_ = self.decoder_.num_decoded_pkts_
        # performance metric
        match_success_count = 0
        not_match_idx = []
        for i in self.get_decoded_pkt_idx(graph_to_decode):
            if np.sum(self.dout_[i] != self.data_[:,i]) == 0:
                match_success_count += 1
            else:
                not_match_idx.append(i)
                # print(self.dout_[i])
        if silent == False:
            if len(not_match_idx) == 0:
                print('[bats_box] all decoded packets match\n\n')
            else:
                print('[bats_box] %d/%d packets not match.\nwith index:'%(len(not_match_idx),len(self.get_decoded_pkt_idx(graph_to_decode))))
                print(not_match_idx,'\n')

        # return metrics
        return (self.eval_metric(), self.get_decoded_pkt_idx(graph_to_decode))
        
    def run_avg(self, num_hops=1,loss_rate=0.1,inactivation=False,silent=True,return_dpkt=False):
        num_tests = 10
        res = [self.run(num_hops,loss_rate,inactivation,silent) for _ in range(num_tests)]
        ar = np.mean([i[0][0] for i in res])
        tr = np.mean([i[0][1] for i in res])
        self.pkt_per_edge_ = np.mean([i[0][2] for i in res])
        dpkt = []
        for i in res:
            dpkt += i[1]
        if return_dpkt:
            return ar, tr, dpkt
        return ar, tr

    def eval_metric(self):
        achieved_code_rate = self.num_decoded_pkts_ / (self.batch_num_*self.bsize_)
        target_code_rate = self.decoder_.total_num_vnodes_ / (self.batch_num_*self.bsize_)
        pkt_per_edge = self.num_decoded_pkts_ / len(self.graph_.edges_)
        self.pkt_per_edge_ = pkt_per_edge
        return achieved_code_rate, target_code_rate, pkt_per_edge
