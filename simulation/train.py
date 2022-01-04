import sys, os
sys.path.insert(1,os.getcwd())
sys.path.insert(1,'../bats-box')
sys.path.insert(1,'.')
import numpy as np
import utils as ut
import batscode as bats
import matplotlib.pyplot as plt
import rl.models as md
import rl.trainer as trainer
import torch
import torch.nn as nn
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='args parse')

parser.add_argument('--n_pkt', dest='n_pkt', type=int, default=64, help='number of input packets')
parser.add_argument('--n_batch', dest='n_batch', type=int, default=10, help='number of generated batches')
parser.add_argument('--pk_size', dest='pk_size', type=int, default=256, help='size of each packet')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='size of the batch')
parser.add_argument('--num_hops', dest='num_hops', type=int, default=10, help='number of hops')
parser.add_argument('--loss_rate', dest='loss_rate', type=float, default=0.1, help='loss rate')

args = parser.parse_args()

n_pkt = args.n_pkt
n_batch = args.n_batch
pk_size = args.pk_size
batch_size = args.batch_size
state_size = 128
num_hops = args.num_hops
loss_rate = args.loss_rate

trainer.write('Training Configuration: n_pkt %d, n_batch %d, batch_size %d, num_hops %d, loss_rate %.3f'%(n_pkt,n_batch, batch_size,num_hops,loss_rate))
print('Training Configuration: n_pkt %d, n_batch %d, batch_size %d, num_hops %d, loss_rate %.3f'%(n_pkt,n_batch, batch_size,num_hops,loss_rate))
agent = md.DDPG(n_pkt, n_batch, pk_size, batch_size, num_hops=num_hops,loss_rate=loss_rate, state_size=state_size, mode = 'cnn',replay_buffer_size=10000, parallel=True)

trainer.train_agent(agent,num_hops=num_hops,loss_rate=loss_rate,max_episode=500,perb=0.02)
agent.save()
reward_list, info_list, pke_list = trainer.test_agent(agent,20)

trainer.write((np.mean(reward_list), np.mean(info_list), np.mean(pke_list)))

plt.figure()
plt.plot(agent.actor_loss_list,label='actor_loss')
plt.plot(agent.critic_judge_list,label='critic_judge')
plt.legend()
plt.title('Actor loss for training 500 epochs')
plt.savefig('actor_loss.jpg',dpi=400)

plt.figure()
plt.plot(agent.critic_loss_list,label='critic loss')
plt.legend()
plt.title('Critic loss for training 500 epochs')
plt.savefig('critic_loss.jpg',dpi=400)


