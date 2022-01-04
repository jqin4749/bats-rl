import sys, os, time
__curdir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1,os.path.join(__curdir__,'../../bats-box'))
sys.path.insert(1,__curdir__)
import torch
import torch.nn as nn
import numpy as np
import batscode as bats
import utils as ut
import models as md

def write(content,name='output.txt'):
    with open(name, 'a+') as f:
        print(content,file=f)
def write_action(act,name='output.txt', mod='int'):
    with open(name, 'a+') as f:
        print('------------------',file=f)
        for row in act:
            print('[',end='',file=f)
            for col in row:
                if mod == 'int':
                    print("%d,"%col,end='',file=f)
                else:
                    print("%.4f,"%col,end='',file=f)
            print('],\n',end='',file=f)

def cut_off(x):
    if x < 0:
        return 0
    elif x > 1:
        return 1
    else:
        return x

def reorder(m):
    getd = lambda x: x[0]
    m_to_return = np.zeros((m.shape[0],m.shape[1]),dtype=int)
    arr = [(d, idx) for idx, d in enumerate(np.sum(m,axis=1))]
    arr.sort(key=getd)
    for ii, (d, idx) in enumerate(arr):
        m_to_return[ii] = m[idx]
    # np.random.shuffle(m_to_return[:int(m.shape[0]/2)])
    return m_to_return


def saturate_action(action,perb = 0.01):
    # action: (n_vns)
    ac = np.zeros(action.shape)
    for idx, i in enumerate(action):
        prop = cut_off(i + np.random.normal(0,perb,size=1)[0])
        # if perb != 0:
        # ac[idx] = np.random.choice([0,1],1,p=[1-prop,prop])[0]
        # else:
        ac[idx] = 1 if prop >= 0.5 else 0
        action[idx] = prop
    if np.sum(ac) == 0:
        ac[np.argmax(action)] = 1
    return ac

def saturate_actions(actions,perb = 0.01):
    # actions: (n_cns, n_vns)
    ac = np.concatenate([np.expand_dims(saturate_action(i,perb),axis=0) for i in actions.T],axis=0).T
    for i, o in zip(ac, actions):
        if np.sum(i) == 0:
            i[np.argmax(o)] = 1
    return ac


def train_agent(agent:md.DDPG,num_hops=50,loss_rate=0.1,max_episode=1000,perb=0.01, output_file_name = 'output.txt'):
    n_vns = agent.n_vns_
    n_cns = agent.n_cns_ # max n_cns
    pk_size = agent.pk_size_
    bsize = agent.bsize_
    steps = 10
    counter = 0
    total_reward = 0
    # env = md.bats_env(n_vns,n_cns,pk_size,bsize,num_hops,loss_rate)
    if len(agent.replay_buffer.storage_) < 100:
        agent.gen_guiding_samples(200)
        agent.update_critic()
        agent.update_actor()
    
    for ep in range(1, max_episode):
        
        info_list = []
        reward_list = []
        action_list = []
        action_raw_list = []
        pke_list = []
        state_list = []
        start = time.time()
        
        for _ in range(steps):
            state = agent.gen_states()
            action_raw = agent.gen_action(state)
            action = saturate_actions(action_raw,perb) # action: (n_cns, n_vns)
            action, action_raw = md.reorder_pair(action, action_raw)
            action_list.append(action)
            action_raw_list.append(action_raw)
            state_list.append(state)
        if agent.parallel_:
            reward_list, info_list, pke_list  = md.run_steps_threading(agent,np.concatenate([np.expand_dims(i,axis=0) for i in action_list],axis=0),
                                                            np.concatenate([np.expand_dims(i,axis=0) for i in action_raw_list],axis=0), state_list)
        else:
            reward_list, info_list, pke_list  = md.run_steps(agent,np.concatenate([np.expand_dims(i,axis=0) for i in action_list],axis=0),
                                                            np.concatenate([np.expand_dims(i,axis=0) for i in action_raw_list],axis=0), state_list)

        if total_reward > 0 and np.sum(reward_list):
            counter += 1
     
        total_reward = np.sum(reward_list)
        end = time.time()
        agent.reward_history_.append(np.mean(reward_list))
        agent.try_save()
        #####
        # if counter > 4:
        #     steps = 6
        if counter == 0 and ep%100 == 0:
            agent.gen_guiding_samples(100)
        #####
        strings = 'Ep:%d (time:%d s) TR: %f '%(ep,(end-start), total_reward)
        strings += 'AR:'
        strings += str(['{0:0.2f}'.format(i) for i in info_list])
        strings += 'PK/E:'
        strings += str(['{0:0.2f}'.format(i) for i in pke_list])
        strings += 'RW:'
        strings += str(['{0:0.2f}'.format(i) for i in reward_list])
        print(strings)
        write(strings,output_file_name)
        for i in action_list:
            write_action(i,output_file_name)
        write('\n',output_file_name)
            
        # if ep == int(max_episode/2):
        #     agent.save()
        if ep % 50 == 0:
            reward_list, info_list, pke_list = test_agent(agent,10)
            write((np.mean(reward_list), np.mean(info_list), np.mean(pke_list)))
            agent.save()
        agent.update()
    return 


def test_agent(agent:md.DDPG,max_steps=400,return_dpkt=False):
    
    total_reward = 0
    action_list = []
    action_raw_list = []
    reward_list = []
    info_list = []
    pke_list = []
    state_list = []

    for stps in range(max_steps):
        state = agent.gen_states()
        action_raw = agent.gen_action(state)
        action = saturate_actions(action_raw,perb=0.0)
        action, action_raw = md.reorder_pair(action, action_raw)
        action_list.append(action)
        action_raw_list.append(action_raw)
        state_list.append(state)
    action_list = np.concatenate([np.expand_dims(i,axis=0) for i in action_list],axis=0)
    if agent.parallel_:
        if return_dpkt:
            reward_list, info_list, pke_list, dpkts  = md.run_steps_threading(agent,action_list,action_list,state_list,return_dpkt)
        else:  
            reward_list, info_list, pke_list  = md.run_steps_threading(agent,action_list,action_list,state_list,return_dpkt)
    else:
        reward_list, info_list, pke_list  = md.run_steps(agent,action_list,action_list,state_list)
    total_reward = np.sum(reward_list)

    strings = 'TR: %f '%(total_reward)
    strings += 'AR:'
    strings += str(['{0:0.2f}'.format(i) for i in info_list])
    strings += 'PK/E:'
    strings += str(['{0:0.2f}'.format(i) for i in pke_list])
    strings += 'RW:'
    strings += str(['{0:0.2f}'.format(i) for i in reward_list])
    print(strings)
    write('[Testing] Results','test_output.txt')
    write(strings,'test_output.txt')
    for i in action_list:
            write_action(i,'test_output.txt')
    write('\n','test_output.txt')
    for i in action_raw_list:
            write_action(i,'test_output.txt','float')
    if return_dpkt:
        return reward_list, info_list, pke_list, dpkts
        
    return reward_list, info_list, pke_list
    

def gen_action(agent:md.DDPG):
    return
