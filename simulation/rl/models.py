
import sys, os, time
__curdir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1,os.path.join(__curdir__,'../../bats-box'))
sys.path.insert(1,__curdir__)
import torch
import torch.nn as nn
import numpy as np
import batscode as bats
import utils as ut
import multiprocessing as mp
# from multiprocessing.pool import ThreadPool
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import shutil

class actor_cnn(nn.Module):
    def __init__(self,n_vns, n_cns, state_size = 128) -> None:
        super(actor_cnn,self).__init__()
        self.n_vns_ = n_vns
        self.n_cns_ = n_cns
        self.state_size_ = state_size
        self.hidden_c_ = 64
        self.filter_size_ = 2
        self.num_cnn_layer_ = 5
        self.padding_ = 1

        self.cnn_dim1_ = self.get_conv2d_trans_input_size(n_cns)
        self.cnn_dim2_ = self.get_conv2d_trans_input_size(n_vns)

        self.input_layers = nn.Sequential(
            nn.Linear(self.state_size_, self.hidden_c_*self.cnn_dim1_*self.cnn_dim2_),
            nn.BatchNorm1d(self.hidden_c_*self.cnn_dim1_*self.cnn_dim2_),
            nn.LeakyReLU()
        )

        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_c_,self.hidden_c_*8, self.filter_size_,padding=self.padding_),
            nn.BatchNorm2d(self.hidden_c_*8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.hidden_c_*8,self.hidden_c_*4, self.filter_size_,padding=self.padding_),
            nn.BatchNorm2d(self.hidden_c_*4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.hidden_c_*4,self.hidden_c_*2, self.filter_size_,padding=self.padding_),
            nn.BatchNorm2d(self.hidden_c_*2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.hidden_c_*2,self.hidden_c_, self.filter_size_,padding=self.padding_),
            nn.BatchNorm2d(self.hidden_c_),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.hidden_c_,1, self.filter_size_,padding=self.padding_),
            nn.Sigmoid()
        )

    def get_conv2d_trans_input_size(self, ts):
        d = ts
        for i in range(self.num_cnn_layer_):
            d = self.cnn_dim(d)
        return d
        
    def cnn_dim(self,ts):
        return ts + 2 * self.padding_ - self.filter_size_ + 1

    def forward(self,x):
        assert x.ndim == 2, '[actor_cnn] ndim error. got %d'%x.ndim
        assert x.shape[-1] == self.state_size_
        out = self.input_layers(x)
        out = out.view(-1, self.hidden_c_, self.cnn_dim1_, self.cnn_dim2_)
        out = self.cnn(out)
        assert out.shape[1] == 1 and out.shape[2] == self.n_cns_ and out.shape[3] == self.n_vns_
        return out

class actor_fcn(nn.Module):
    def __init__(self,n_vns, n_cns, state_size = 128) -> None:
        super(actor_fcn,self).__init__()
        self.n_vns_ = n_vns
        self.n_cns_ = n_cns
        self.state_size_ = state_size
        self.hidden_c_ = 16
        self.hidden_linear_ = 16
        self.filter_size_ = 2
        self.num_cnn_layer_ = 5
        self.pert_rate_ = 0.2
        self.padding_ = 1
        self.dev_ =  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.cnn_dim_ = self.get_conv1d_trans_input_size(self.hidden_linear_)

        self.input_layers = nn.Sequential(
            nn.Linear(self.state_size_, self.hidden_c_*self.cnn_dim_),
            nn.BatchNorm1d(self.hidden_c_*self.cnn_dim_),
            nn.LeakyReLU()
        )

        self.cnn = nn.Sequential(
            nn.ConvTranspose1d(self.hidden_c_,self.hidden_c_*8, self.filter_size_,padding=self.padding_),
            nn.BatchNorm1d(self.hidden_c_*8),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(self.hidden_c_*8,self.hidden_c_*4, self.filter_size_,padding=self.padding_),
            nn.BatchNorm1d(self.hidden_c_*4),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(self.hidden_c_*4,self.hidden_c_*2, self.filter_size_,padding=self.padding_),
            nn.BatchNorm1d(self.hidden_c_*2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(self.hidden_c_*2,self.hidden_c_, self.filter_size_,padding=self.padding_),
            nn.BatchNorm1d(self.hidden_c_),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(self.hidden_c_, 1, self.filter_size_, padding=self.padding_),
            nn.LeakyReLU()
        )

        self.output_layers = nn.Sequential(
            nn.Linear(self.hidden_linear_, self.n_cns_*self.n_vns_),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        assert x.ndim == 2, '[actor_fcn] ndim error. got %d'%x.ndim
        assert x.shape[-1] == self.state_size_
        out = self.input_layers(x)
        out = out.contiguous().view(-1, self.hidden_c_, self.cnn_dim_)
        out = self.cnn(out)
        # out = torch.unsqueeze(out,dim=1)
        out = out.contiguous().view(-1, self.hidden_linear_)
        # if self.training:
        #     idx = np.random.choice(out.shape[0], int(out.shape[0] * self.pert_rate_)) 
        #     out[idx,:] += torch.rand(self.hidden_linear_).to(self.dev_)
        out = self.output_layers(out).contiguous().view(-1, 1, self.n_cns_, self.n_vns_)
        assert out.shape[1] == 1 and out.shape[2] == self.n_cns_ and out.shape[3] == self.n_vns_
        return out

    def get_conv1d_trans_input_size(self,target):
        d = target
        for i in range(self.num_cnn_layer_):
            d = self.cnn_dim(d)
        return d

    def cnn_dim(self,ts):
        return ts + 2 * self.padding_ - self.filter_size_ + 1

class critic_cnn(nn.Module):
    def __init__(self,n_vns, n_cns) -> None:
        super(critic_cnn,self).__init__()
        self.n_vns_ = n_vns
        self.n_cns_ = n_cns
        self.hidden_c_ = 32
        self.filter_size_ = 3
        self.padding_ = 1
        self.cnn = nn.Sequential(
            nn.Conv2d(1,self.hidden_c_*8,self.filter_size_ , padding=self.padding_),
            nn.BatchNorm2d(self.hidden_c_*8),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_c_*8,self.hidden_c_*4,self.filter_size_ ,padding=self.padding_ ),
            nn.BatchNorm2d(self.hidden_c_*4),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_c_*4,self.hidden_c_*2,self.filter_size_,padding=self.padding_),
            nn.BatchNorm2d(self.hidden_c_*2),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_c_*2,self.hidden_c_,self.filter_size_ ,padding=self.padding_ ),
            nn.BatchNorm2d(self.hidden_c_),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_c_,self.hidden_c_,self.filter_size_ ,padding=self.padding_ ),
            nn.BatchNorm2d(self.hidden_c_),
            nn.LeakyReLU()
        )

        self.linear_input_size = self.get_conv_out_size()
        self.linears = nn.Linear(self.linear_input_size + self.n_cns_ + self.n_vns_, 1) 

    def forward(self,x):  # x: action (-1, 1, n_cns, n_vns)
        if x.ndim == 3:
            x = torch.unsqueeze(x,dim=1)
        assert x.ndim == 4, '[critic_cnn] ndim error. got %d'%x.ndim
        ds = (torch.sum(torch.round(x),dim=-1) / self.n_vns_).view(-1,self.n_cns_)
        ns = (torch.sum(torch.round(x),dim=-2) / self.n_cns_).view(-1,self.n_vns_)
        out = self.cnn(x) # state 
        out = out.contiguous().view(-1,self.linear_input_size)
        out = torch.cat((out, ds, ns),dim=-1)
        out = self.linears(out)
        return out

    def get_conv_out_size(self):
        t = torch.rand(1,1,self.n_cns_,self.n_vns_) 
        with torch.no_grad():
            o = self.cnn(t)
        o = o.view(1,-1)
        return o.shape[-1]


class critic_fcn(nn.Module):
    def __init__(self,n_vns, n_cns) -> None:
        super(critic_fcn,self).__init__()
        self.n_vns_ = n_vns
        self.n_cns_ = n_cns
        self.hidden_c_ = 16
        self.hidden_c__linear_ = 64
        self.filter_size_ = 3
        self.padding_ = 1

        self.input_layer = nn.Sequential(
            nn.Linear(self.n_vns_*self.n_cns_,self.hidden_c__linear_),
            nn.BatchNorm1d(self.hidden_c__linear_),
            nn.LeakyReLU()
        )
        
        self.fcn = nn.Sequential(
            nn.Conv1d(1,self.hidden_c_,self.filter_size_ , padding=self.padding_),
            nn.BatchNorm1d(self.hidden_c_),
            nn.LeakyReLU(),
            nn.Conv1d(self.hidden_c_,self.hidden_c_*2,self.filter_size_ , padding=self.padding_),
            nn.BatchNorm1d(self.hidden_c_*2),
            nn.LeakyReLU(),
            nn.Conv1d(self.hidden_c_*2,self.hidden_c_*4,self.filter_size_ , padding=self.padding_),
            nn.BatchNorm1d(self.hidden_c_*4),
            nn.LeakyReLU(),
            nn.Conv1d(self.hidden_c_*4,self.hidden_c_*2,self.filter_size_ , padding=self.padding_),
            nn.BatchNorm1d(self.hidden_c_*2),
            nn.LeakyReLU(),
            nn.Conv1d(self.hidden_c_*2,self.hidden_c_,self.filter_size_ , padding=self.padding_),
            nn.BatchNorm1d(self.hidden_c_),
            nn.LeakyReLU()
        )
        self.linear_input_size = self.get_conv_out_size()
        self.out_layer = nn.Linear(self.linear_input_size + self.n_cns_ + self.n_vns_, 1) 

    def forward(self,x):  # x: action (-1, 1, n_cns, n_vns)
        if x.ndim == 3:
            x = torch.unsqueeze(x,dim=1)
        assert x.ndim == 4, '[critic_cnn] ndim error. got %d'%x.ndim
        ds = (torch.sum(x,dim=-1) / self.n_vns_).view(-1,self.n_cns_)
        ns = (torch.sum(x,dim=-2) / self.n_cns_).view(-1,self.n_vns_)
        x = torch.squeeze(x,dim=1).contiguous().view(-1, self.n_cns_*self.n_vns_) # to compensate the input format due to cnn
        out = self.input_layer(x) 
        out = self.fcn(torch.unsqueeze(out,dim=1)).contiguous().view(-1,self.linear_input_size)
        out = torch.cat((out, ds, ns),dim=-1)
        out = self.out_layer(out)
        return out

    def get_conv_out_size(self):
        t = torch.rand(1, 1, self.hidden_c__linear_) 
        with torch.no_grad():
            o = self.fcn(t)
        o = o.view(1,-1)
        return o.shape[-1]
   

class replay_buffer:
    def __init__(self,max_size = 100, parallel = False) -> None:
        # thread safe
        self.storage_ = {}
        self.max_size_ = max_size
        self.ptr_ = 0
        # if parallel:
        #     self.lock_ = mp.Lock()
        self.parallel_ = parallel

    def push(self,data):
        # Expects tuples of (state, action, reward, done, info)
        # if self.parallel_:
        #     self.lock_.acquire()

        self.storage_[self.ptr_] = data
        self.ptr_ = (self.ptr_ + 1) % self.max_size_
        # if self.parallel_:
        #     self.lock_.release()

    def sample(self, bs):
        assert len(self.storage_) != 0, '[replay_buffer] cannot sample an empty buffer'
        
        idx = np.random.choice(len(self.storage_),min(bs,len(self.storage_)))
        ind = [self.storage_[i] for i in idx]   
        x, u, r, d, v = [], [], [], [], []
        for i in ind:
            X, U, R, D, V = i
            x.append(X)
            u.append(U)
            r.append(R)
            d.append(D)
            v.append(V)

        x = np.concatenate([np.expand_dims(i,axis=0) for i in x],axis=0)
        u = np.concatenate([np.expand_dims(i,axis=0) for i in u],axis=0)
        r = np.concatenate([np.expand_dims(i,axis=0) for i in r],axis=0)
        d = np.concatenate([np.expand_dims(i,axis=0) for i in d],axis=0)
        v = np.concatenate([np.expand_dims(i,axis=0) for i in v],axis=0)

        return x, u, r, d, v

    def check_need_for_clean(self):
        if len(self.storage_) == 0:
            return False
        zero_count = 0
        for k, v in self.storage_.items():
            if v[2] == 0:
                zero_count += 1
        if zero_count/len(self.storage_) > 0.1:
            return True
        return  False

    

def gen_decodable_batch(n_cns,n_vns):
    arr_to_return = np.zeros((n_cns,n_vns),dtype=int)
    available_cns = list(np.arange(n_cns,dtype=int))
    for i in range(n_vns):
        if len(available_cns) == 0:
            available_cns = list(np.arange(n_cns,dtype=int))
        idx = np.random.choice(available_cns,1)[0]
        available_cns.remove(idx)
        arr_to_return[idx,i] = 1

    return arr_to_return

def reverse(v):
    # (nbatch, nvns)
    arr_to_return = np.zeros(v.shape)
    for row, i in enumerate(v):
        for col, j in enumerate(i):
            if j == 1:
                arr_to_return[row][col] = 0
            else:
                arr_to_return[row][col] = 1
    return arr_to_return

def get1():
    while 1:
        a = np.random.rand()
        if a > 0.5 and a <= 1:
            return a
def get0():
    while 1:
        a = np.random.rand()
        if a <= 0.5 and a >= 0:
            return a

def to_ratio(m):
    m_to_return = np.zeros((m.shape[0],m.shape[1]))
    for row, i in enumerate(m):
        for col, j in enumerate(i):
            if j == 1:
                m_to_return[row][col] = get1()
            else:
                m_to_return[row][col] = get0()
    return m_to_return

def reorder(m):
    getd = lambda x: x[0]
    m_to_return = np.zeros((m.shape[0],m.shape[1]))
    arr = [(d, idx) for idx, d in enumerate(np.sum(m,axis=1))]
    arr.sort(key=getd)
    for ii, (d, idx) in enumerate(arr):
        m_to_return[ii] = m[idx]
    return m_to_return

def reorder_pair(m, m_raw):
    getd = lambda x: x[0]
    m_to_return = np.zeros((m.shape[0],m.shape[1]))
    m_to_return_raw = np.zeros((m.shape[0],m.shape[1]))
    arr = [(d, idx) for idx, d in enumerate(np.sum(m,axis=1))]
    arr.sort(key=getd)
    for ii, (d, idx) in enumerate(arr):
        m_to_return[ii] = m[idx]
        m_to_return_raw[ii] = m_raw[idx]
    return m_to_return, m_to_return_raw


class DDPG:
    def __init__(self,n_vns, n_cns, pk_size, bsize, num_hops, loss_rate=0.1, state_size = 128, 
                                                    mode = 'cnn', replay_buffer_size = 100, parallel = False) -> None:
        self.n_vns_ = n_vns
        self.n_cns_ = n_cns
        self.pk_size_ = pk_size
        self.bsize_ = bsize
        self.num_hops_ = num_hops
        self.loss_rate_ = loss_rate
        self.state_size_ = state_size
        self.gd_ = 0.0
        self.dev_ =  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.actor = actor_cnn(n_vns, n_cns, state_size) # if mode == 'cnn' else actor_fcn(n_vns, n_cns,state_size) 
        self.actor_opt = torch.optim.Adam(self.actor.parameters(),lr=2e-4)

        self.critic = critic_cnn(n_vns, n_cns) if mode == 'cnn' else critic_fcn(n_vns, n_cns) 
        self.critic_opt = torch.optim.Adam(self.critic.parameters(),lr=1e-3)
        
        self.criterion = nn.SmoothL1Loss()
        self.mse = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.replay_buffer = replay_buffer(replay_buffer_size, parallel)
        self.actor.to(self.dev_)
        self.critic.to(self.dev_)

        self.actor_loss_list = []
        self.critic_loss_list = []
        self.guided_loss_list = []
        self.critic_judge_list = []
        self.last_action_diff_list = []
        self.parallel_ = parallel
        self.mode_ = mode
        self.episode_train_ = 0
        self.reward_history_ = []
        # create directory to save running checkpoints
        running_dir = os.getcwd()
        if not os.path.isdir(os.path.join(running_dir,'checkpoints')):
            os.makedirs('checkpoints', exist_ok = True)
        date_time = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
        cp_dir = os.path.join(running_dir,'checkpoints','job-%s'%date_time)
        os.makedirs(cp_dir, exist_ok = True)
        self.job_dir_ = cp_dir

    def gen_action(self, state):
        if state.ndim == 1:
            state = np.expand_dims(state,axis=0)
        state = torch.FloatTensor(state).to(self.dev_)
        self.actor.eval()
        return self.actor(state).detach().cpu().numpy().reshape(self.n_cns_, self.n_vns_)

    def gen_states(self):
        return np.random.normal(0,1.0,size=self.state_size_)

    def get_loss(self):
        return self.actor_loss_list, self.critic_loss_list

    def gen_decodable_batches(self, n_cns, n_vns, bnum):
        actions = []
        for _ in range(int(bnum)):
                # actions.append(gen_decodable_batch(n_cns,n_vns))
                actions.append(reorder(ut.gen_binary_graph_from_dist(n_vns,n_cns,'./K%dM%d.txt'%(n_vns,self.bsize_))))

        # return np.concatenate([np.expand_dims(gen_decodable_batch(n_cns, n_vns),axis=0) for _ in range(bnum)],axis=0)
        return np.concatenate([np.expand_dims(i,axis=0) for i in actions],axis=0)

    def update_critic(self):
        bs = 32 # default
        epochs = int(np.ceil(len(self.replay_buffer.storage_) / bs) + 1) * 100
        critic_loss_list = []
        print('pretrain the critic')
        for epoch in range(epochs):
            # Sample replay buffer
            x, u, r, d, v = self.replay_buffer.sample(bs)
            state = torch.FloatTensor(x).to(self.dev_)
            action = torch.FloatTensor(u).to(self.dev_)
            # done = torch.unsqueeze(torch.FloatTensor(1-d).to(self.dev_),dim=-1)
            reward = torch.unsqueeze(torch.FloatTensor(r).to(self.dev_),dim=-1)
            # Compute the target Q value
            target_Q = reward
            # Get current Q estimate
            self.critic.train()
            current_Q = self.critic(action)
            # Compute critic loss
            critic_loss = self.mse(current_Q, target_Q)
            critic_loss_list.append(critic_loss.item())
            # Optimize the critic
            self.critic_opt.zero_grad()
            critic_loss.backward()
            # nn.utils.clip_grad_norm_(self.critic.parametsers(), 1.0)
            self.critic_opt.step()
        
        plt.figure()
        plt.plot(critic_loss_list)
        plt.title('critic pretrain loss')
        plt.savefig('critic_pretrain.jpg',dpi=400)   

    def update_actor(self):
        bs = 32 # default
        critic_judge_list = []
        epochs = int(np.ceil(len(self.replay_buffer.storage_) / bs) + 1) * 100
        print('pretrain the actor')
        for epoch in range(epochs):
            # Sample replay buffer
            x, u, r, d, v = self.replay_buffer.sample(bs)
            state = torch.FloatTensor(x).to(self.dev_)
            action = torch.FloatTensor(u).to(self.dev_)
            reward = torch.unsqueeze(torch.FloatTensor(r).to(self.dev_),dim=-1)
            
            # Compute actor loss
            # Optimize the actor
            self.actor.train()
            act = self.actor(state)
            critic_judge = self.critic(act).mean()
            actor_loss = -critic_judge 

            self.actor_opt.zero_grad()
            actor_loss.backward()
            # nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()
            critic_judge_list.append(critic_judge.item())
        plt.figure()
        plt.plot(critic_judge_list)
        plt.title('critic judge loss')
        plt.savefig('critic_judge_loss.jpg',dpi=400)   


    def update(self):
        bs = 16 # default
        factor = 2
        if self.episode_train_ >= 100:
            factor = 1
        self.episode_train_ += 1
        epochs = int(np.ceil(len(self.replay_buffer.storage_) / bs) + 1) * factor
        for epoch in range(epochs):
            # Sample replay buffer
            x, u, r, d, v = self.replay_buffer.sample(bs)
            state = torch.FloatTensor(x).to(self.dev_)
            action = torch.FloatTensor(u).to(self.dev_)
            # done = torch.unsqueeze(torch.FloatTensor(1-d).to(self.dev_),dim=-1)
            reward = torch.unsqueeze(torch.FloatTensor(r).to(self.dev_),dim=-1)
            # Compute the target Q value

            target_Q = reward
            # Get current Q estimate
            self.critic.train()
            current_Q = self.critic(action)
    
            # Compute critic loss
            critic_loss = self.mse(current_Q, target_Q)
            self.critic_loss_list.append(critic_loss.item())
            # Optimize the critic
            self.critic_opt.zero_grad()
            critic_loss.backward()
            # nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_opt.step()

            # Compute actor loss
           
            # Optimize the actor
            self.actor.train()
            act = self.actor(state)
            # batch_guide = torch.sum(torch.FloatTensor(self.gen_decodable_batches(self.n_cns_,self.n_vns_,act.shape[0]))).to(self.dev_)
            # batch_guide = torch.FloatTensor(np.concatenate([np.expand_dims(self.batch_guide_,axis=0) for _ in range(act.shape[0])],axis=0)).to(self.dev_)
            # guided_loss = (self.gd_) * self.criterion(torch.sum(torch.round(torch.squeeze(act,dim=1))), batch_guide)

            critic_judge = self.critic(act).mean()
            actor_loss = -critic_judge 

            self.actor_opt.zero_grad()
            actor_loss.backward()
            # nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()

            self.actor_loss_list.append(actor_loss.item())
            # self.guided_loss_list.append(guided_loss.item())
            self.critic_judge_list.append(critic_judge.item())

    def gen_guiding_samples(self,num_samples = 50):
        # use gen_decodable_batches() and preset_distribution to genenerate 100 guiding samples
        # multi threads enabled
        print('[DDPG] Generating guiding samples')
        start = time.time()
        actions_raw = []
        actions = []
        state_list = []
        for _ in range(num_samples):
            actions.append(gen_decodable_batch(self.n_cns_,self.n_vns_))
            state_list.append(self.gen_states())
            actions.append(ut.gen_binary_graph_from_dist_all_connected(self.n_vns_,self.n_cns_,'./K%dM%d.txt'%(self.n_vns_,self.bsize_)))
            state_list.append(self.gen_states())

        actions = np.concatenate([np.expand_dims(reorder(i),axis=0) for i in actions],axis=0)
        actions_raw = np.concatenate([np.expand_dims(to_ratio(i),axis=0) for i in actions],axis=0)
        # actions_raw = np.concatenate([np.expand_dims(i,axis=0) for i in actions_raw],axis=0)
        if self.parallel_:
            reward_list, info_list, _ = run_steps_threading(self, actions, actions_raw, state_list)    
        else:
            reward_list, info_list, _ = run_steps(self, actions,actions_raw, state_list)
        end = time.time()
        print('[DDPG] Finished. Time elapsed: %.4f min'%((end-start)/60.0))
        return reward_list, info_list
        
    def save(self, dir_name = 'running_states'):
        
        running_dir = os.getcwd()
        if not os.path.isdir(os.path.join(running_dir,dir_name)):
            os.makedirs(dir_name, exist_ok = True)
        date_time = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
        cp_dir = os.path.join(running_dir,dir_name,date_time)
        os.makedirs(cp_dir, exist_ok = True)
        info = {}
        info['n_vns'] = self.n_vns_ 
        info['n_cns'] = self.n_cns_ 
        info['pk_size'] = self.pk_size_ 
        info['bsize'] = self.bsize_ 
        info['num_hops'] = self.num_hops_ 
        info['loss_rate'] = self.loss_rate_ 
        info['state_size'] = self.state_size_ 
        info['mode'] = self.mode_ 
        info['episode_train_'] = self.episode_train_
        info['replay_max_size'] = self.replay_buffer.max_size_
        info['replay_ptr'] = self.replay_buffer.ptr_
        # print README.md
        with open(os.path.join(cp_dir,'README.md'),'w+') as f:
            print('Check point saved at %s'%date_time,file=f)
            for k,v in info.items():
                print('%s: '%k,v, file=f)
            print('Average rewards achieved: %f'%self.reward_history_[-1], file=f)
        with open(os.path.join(cp_dir,'info_dict.pkl'),'wb') as f:
            pickle.dump(info,f)
        with open(os.path.join(cp_dir,'replay_buffer.pkl'),'wb') as f:
            pickle.dump(self.replay_buffer.storage_,f)
        np.save(os.path.join(cp_dir,'actor_loss_list.npy'),self.actor_loss_list)
        np.save(os.path.join(cp_dir,'critic_judge_list.npy'),self.critic_judge_list)
        np.save(os.path.join(cp_dir,'critic_loss_list.npy'),self.critic_loss_list)
        # np.save(os.path.join(cp_dir,'guided_loss_list.npy'),self.guided_loss_list)
        np.save(os.path.join(cp_dir,'reward_history.npy'),self.reward_history_)
        torch.save(self.actor, os.path.join(cp_dir,'actorpara.pth'))
        torch.save(self.critic, os.path.join(cp_dir,'criticpara.pth'))
        os.system('cp %s %s'%(os.path.join(running_dir,'check_loss_template.ipynb'),cp_dir))

    def try_save(self):
        '''dynamic save check points with highest testing accuracy'''
        getr = lambda x: x[1]
        max_checkpoints = 10
        # get rewards of saved checkpoints
        rs = []
        names = []
        for d in os.listdir(self.job_dir_):
            try:
                r = np.load(os.path.join(self.job_dir_, d, 'reward_history.npy'))
                names.append(d)
            except:
                continue
            rs.append(r[-1])
        # decided whether need to save the current one
        if len(rs) != 0:
            if self.reward_history_[-1] < min(rs): return
            if len(rs) == max_checkpoints:
                # remove the existing checkpoints
                ids = [(idx, i) for idx, i in enumerate(rs) if i <= self.reward_history_[-1]]
                ids.sort(key=getr)
                idx_to_remove = ids[0][0]
                shutil.rmtree(os.path.join(self.job_dir_, names[idx_to_remove]))

        # save the current one
        self.save(dir_name = self.job_dir_)
        return 

    def load(self, name=None, dir_name = 'running_states'):
        # load model with the same configure as the current one
        info_ = {}
        info_['n_vns'] = self.n_vns_ 
        info_['n_cns'] = self.n_cns_ 
        info_['pk_size'] = self.pk_size_ 
        info_['bsize'] = self.bsize_ 
        info_['num_hops'] = self.num_hops_ 
        info_['loss_rate'] = self.loss_rate_ 
        info_['state_size'] = self.state_size_ 
        info_['mode'] = self.mode_ 
        running_dir = os.getcwd()
        for d in os.listdir(os.path.join(running_dir,dir_name)):
            dir_ = os.path.join(running_dir,dir_name,d)
            if name != None:
                if d != name:
                    continue
            if not os.path.isdir(dir_): 
                continue
            if d[0] == '.': continue
            try: 
                with open(os.path.join(dir_,'info_dict.pkl'),'rb') as f:
                    info = pickle.load(f)
            except:
                continue
            if not compare_two_dict(info_, info): continue
            
            self.episode_train_ = info['episode_train_']
            self.replay_buffer.max_size_ = info['replay_max_size']
            self.replay_buffer.ptr_ = info['replay_ptr']
            with open(os.path.join(dir_,'replay_buffer.pkl'),'rb') as f:
                    self.replay_buffer.storage_ = pickle.load(f)
            self.actor_loss_list = np.load(os.path.join(dir_,'actor_loss_list.npy'))
            self.critic_judge_list = np.load(os.path.join(dir_,'critic_judge_list.npy'))
            self.critic_loss_list = np.load(os.path.join(dir_,'critic_loss_list.npy'))
            self.reward_history_ = np.load(os.path.join(dir_,'reward_history.npy'))

            self.actor.load_state_dict(torch.load(os.path.join(dir_,'actorpara.pth'),map_location=self.dev_).state_dict())
            self.critic.load_state_dict(torch.load(os.path.join(dir_,'criticpara.pth'),map_location=self.dev_).state_dict())
            return
        print('[DDPG] Fail to find identical checkpoints')

def compare_two_dict(a, b):
    for k , v in a.items():
        if b[k] != v:
            return 0 # not same
    return 1 # same

def run_steps_threading(agent, actions, actions_raw, state_list, return_dpkt = False):
    assert actions.ndim == 3
    agent_info = {}
    agent_info['n_vns_'] = agent.n_vns_
    agent_info['n_cns_'] = agent.n_cns_
    agent_info['pk_size_'] = agent.pk_size_
    agent_info['bsize_'] = agent.bsize_
    agent_info['num_hops_'] = agent.num_hops_
    agent_info['loss_rate_'] = agent.loss_rate_
    agent_info['state_size_'] = agent.state_size_
    agent_info['return_dpkt'] = return_dpkt
    num_workers =  max(mp.cpu_count() - 1, 1) if len(actions) > mp.cpu_count() else len(actions) 
    in_args = [(agent_info, i, i_raw, s) for i, i_raw, s in zip(actions,actions_raw, state_list)]

    with mp.Pool(num_workers) as p:
        out = p.map(single_step, in_args)
    
    reward_list = [i[0] for i in out]
    info_list = [i[1] for i in out]
    pkt_per_edge_list = [i[2] for i in out]
    dpkts = []
    if return_dpkt:
        for i in out:
            dpkts += i[4]
    for i in out:
        data = i[3]
        agent.replay_buffer.push(data)
    if return_dpkt:
        return reward_list, info_list, pkt_per_edge_list, dpkts

    return reward_list, info_list, pkt_per_edge_list

def run_steps(agent, actions, actions_raw, state_list):
    reward_list = []
    info_list = []
    pkt_per_edge_list = []
    env = bats_env(agent.n_vns_, agent.n_cns_, agent.pk_size_, agent.bsize_, agent.num_hops_, agent.loss_rate_)
    for i,i_raw, s in zip(actions,actions_raw, state_list):
                reward, done, info = env.step(i)
                
                agent.replay_buffer.push((s, i_raw, reward, done, info))
                reward_list.append(reward)
                info_list.append(info)
                pkt_per_edge_list.append(env.bats_box_.pkt_per_edge_)
    return reward_list, info_list, pkt_per_edge_list

def single_step(args):
    # n_vns_, n_cns_, pkt_size_, bsize_, num_hops_, loss_rate_, state_size_, replay_buffer, action = args
    agent_info, action, action_raw, s = args
    env = bats_env(agent_info['n_vns_'], agent_info['n_cns_'], agent_info['pk_size_'], agent_info['bsize_'], agent_info['num_hops_'], agent_info['loss_rate_'])
    if agent_info['return_dpkt']:
        reward, done, info, dpkt = env.step(action, return_dpkt = agent_info['return_dpkt'])
    else:
        reward, done, info = env.step(action, return_dpkt = agent_info['return_dpkt'])
    data = (s, action, reward, done, info)
    if agent_info['return_dpkt']:
        return reward, info, env.bats_box_.pkt_per_edge_, data, dpkt

    return reward, info, env.bats_box_.pkt_per_edge_, data

def hops_perturb(hops, max_hops):
    return np.random.binomial(n=max_hops, p=hops/max_hops, size=1)[0]
def loss_rate_perturb(rate):
    return np.random.normal(loc=rate,scale=0.02,size=1)[0]

class bats_env:
    def __init__(self,n_vns, n_cns, pk_size, bsize, num_hops, loss_rate=0.1) -> None:
        global MAX_HOPS 
        MAX_HOPS = 100
        self.bats_box_ = bats.bats_box()
        self.n_vns_ = n_vns
        self.n_cns_ = n_cns # max n_cns
        self.pk_size_ = pk_size
        self.bsize_ = bsize
        self.num_hops_ = num_hops
        self.loss_rate_ = loss_rate
        self.tolerance_ = 0.05
        self.num_hops_p_ = num_hops # hops_perturb(self.num_hops_,MAX_HOPS) 
        self.loss_rate_p_ = loss_rate # loss_rate_perturb(self.loss_rate_)
        
    def step(self,action, silent=True, return_dpkt=False):
        # action: (n_cns, n_vns)
     
        self.graph_ = ut.gen_tanner_from_binary_graph(
                                    action, 
                                    self.pk_size_, self.bsize_)

        self.action_ = action
        self.bats_box_.update_graph(self.graph_)
        if return_dpkt:
            self.achieved_rate_, self.target_rate_, dpkt = self.bats_box_.run_avg(
                                    num_hops=self.num_hops_p_ ,loss_rate=self.loss_rate_p_ ,inactivation=True, silent=silent,return_dpkt=return_dpkt) # interact with bats_box
        else:
            self.achieved_rate_, self.target_rate_ = self.bats_box_.run_avg(
                                num_hops=self.num_hops_p_ ,loss_rate=self.loss_rate_p_ ,inactivation=True, silent=silent,return_dpkt=return_dpkt) # interact with bats_box

        self.done_ = 1 if self.done_condition() else 0
        reward = self.reward_logic()
        if return_dpkt:
            return reward, self.done_, self.achieved_rate_/ self.target_rate_, dpkt
        return reward, self.done_, self.achieved_rate_/ self.target_rate_

    def done_condition(self):
        if self.achieved_rate_ / self.target_rate_ >= 1 - self.tolerance_:
            return True
    def get_num_orphan_vn(self):
        return len([idx for idx, i in enumerate(self.action_.T) if np.sum(i)==0])
    def get_num_orphan_cn(self):
        return len([idx for idx, i in enumerate(self.action_) if np.sum(i)==0])


    def reward_logic(self):
        achieved_rate_reward = self.achieved_rate_ / self.target_rate_
        reward = achieved_rate_reward + self.bats_box_.pkt_per_edge_ + self.get_num_orphan_cn()/self.n_cns_ if self.done_ else achieved_rate_reward
        # reward = reward if self.bats_box_.pkt_per_edge_ > 0.1 else 0.0
        return reward

    def save_state(self,name):
        os.makedirs('./running_states', exist_ok = True)
        np.save('./running_states/%s.npy'%name, self.action_)

    
