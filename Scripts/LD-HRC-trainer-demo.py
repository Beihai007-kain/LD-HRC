import torch
import time
import random
import pandas as pd
from collections import deque
import math
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
from models import SBS_Actor_model, TBS_Actor_model, FAP_Actor_model, SharedCritic_model, GATStateEncoder_model
from env import HRCSMultiAgentEnv
from link_provider import LinkMatrixProvider
from goal_provider import LPValueProvider



class MultiAgentReplayBuffer:

    def __init__(self, max_size, config):

        self.max_size = max_size

        #跟踪下一个经验将存储在缓冲区的位置，
        self.ptr = 0
        self.size = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_sbs, self.num_tbs, self.num_fap = 1, config['num_tbs'], config['num_fap']
        obs_dims = config['obs_dims']

        #总的观察维度
        self.total_obs_dim = (self.num_sbs * obs_dims['sbs'] + self.num_tbs * obs_dims['tbs'] +
                              self.num_fap * obs_dims['fap'])
        sbs_action_dim, tbs_action_dim, fap_action_dim = config['K'], config['W'] * 2, config['N']
        self.total_action_dim = (self.num_sbs * sbs_action_dim + self.num_tbs * tbs_action_dim + self.num_fap *
                                 fap_action_dim)

        self.obs = torch.zeros((max_size, self.total_obs_dim), device=self.device)
        self.next_obs = torch.zeros((max_size, self.total_obs_dim), device=self.device)
        self.actions = torch.zeros((max_size, self.total_action_dim), device=self.device)
        self.rewards = torch.zeros((max_size, 1), device=self.device)
        self.future_returns = torch.zeros((max_size, 1), device=self.device)
        self.dones = torch.zeros((max_size, 1), device=self.device)

    def _flatten_obs(self, obs_dict):
        # 初始化obs_tensors列表，存储观察张量
        obs_tensors = []

        sbs_obs = obs_dict['sbs'][0]
        obs_tensors.append(sbs_obs)

        for i in range(self.num_tbs):
            tbs_obs = obs_dict['tbs'][i]
            obs_tensors.append(tbs_obs)

        for i in range(self.num_fap):
            fap_obs = obs_dict['fap'][i]
            obs_tensors.append(fap_obs)

        # 将所有的观察张量拼接成一个单一的张量
        return torch.cat(obs_tensors)

    # def _flatten_actions(self, actions_dict):
    #     action_tensors = [actions_dict['sbs'][0].flatten()] +
    #     [actions_dict['tbs'][i].flatten() for i in range(self.num_tbs)] +
    #     [actions_dict['fap'][i].flatten() for i in range(self.num_fap)]
    #
    # return torch.cat(action_tensors)

    def _flatten_actions(self, actions_dict):
        # 初始化action_tensors列表，存储动作张量
        action_tensors = []

        sbs_action = actions_dict['sbs'][0].flatten()
        action_tensors.append(sbs_action)

        for i in range(self.num_tbs):
            tbs_action = actions_dict['tbs'][i].flatten()
            action_tensors.append(tbs_action)

        for i in range(self.num_fap):
            fap_action = actions_dict['fap'][i].flatten()
            action_tensors.append(fap_action)

        # 将所有的动作张量拼接成一个单一的张量
        return torch.cat(action_tensors)

    def store(self, obs, actions, reward, next_obs, done):
        self.obs[self.ptr] = self._flatten_obs(obs).to(self.device)
        self.next_obs[self.ptr] = self._flatten_obs(next_obs).to(self.device)
        self.actions[self.ptr] = self._flatten_actions(actions).to(self.device)
        self.rewards[self.ptr] = torch.tensor(reward, device=self.device)
        self.dones[self.ptr] = torch.tensor(done, dtype=torch.float32, device=self.device)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # def sample(self, batch_size):
    #     #打散时间相关性
    #     idxs = np.random.randint(0, self.size, size=batch_size)
    #     return (self.obs[idxs], self.actions[idxs], self.rewards[idxs], self.future_returns[idxs], self.next_obs[idxs],
    #             self.dones[idxs])

    def sample(self, batch_size):
        #安全边界：确保缓冲区非空，并限制 batch
        assert self.size > 0, "Replay buffer is empty."

        batch_size = min(batch_size, self.size)
        idxs = torch.randint(
            low=0, high=self.size, size=(batch_size,), device=self.device, dtype=torch.long
        )

        obs = self.obs.index_select(0, idxs)
        actions = self.actions.index_select(0, idxs)
        rewards = self.rewards.index_select(0, idxs)
        future_return = self.future_returns.index_select(0, idxs)
        next_obs = self.next_obs.index_select(0, idxs)
        dones = self.dones.index_select(0, idxs)

        return obs, actions, rewards, future_return, next_obs, dones

    def post_process_episode(self, start_idx, end_idx, goal_value):
        episode_return = -float(goal_value)

        # 展开环形索引区间
        if start_idx < end_idx:
            indices = list(range(start_idx, end_idx))
        else:
            indices = list(range(start_idx, self.max_size)) + list(range(0, end_idx))

        for idx in indices:
            self.future_returns[idx] = episode_return


# --- 训练器 (Graph-based Actor-Critic) ---
class MADDPG_Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        for key, value in config.items():

            #使用setattr将键值挂成成员属性
            setattr(self, key, value)

        self.max_obs_dim = max(self.obs_dims.values())
        self.gat_out_dim = config['gat_out_dim']

        # 系统中的总节点数
        self.num_nodes = 1 + self.num_tbs + self.num_fap

        self.adj = LinkMatrixProvider.adj

        self.LP_value = LPValueProvider.value

        self.state_encoder = GATStateEncoder_model(
            in_features=self.max_obs_dim, hidden_features=config['gat_hidden_dim'],
            out_features=self.gat_out_dim, n_heads=config['gat_n_heads']
        ).to(self.device)
        self.target_state_encoder = GATStateEncoder_model(
            in_features=self.max_obs_dim, hidden_features=config['gat_hidden_dim'],
            out_features=self.gat_out_dim, n_heads=config['gat_n_heads']
        ).to(self.device)

        #把目标网络权重与主网络对齐，同时还有gat的权重对齐
        self._init_networks()
        self._init_optimizers()
        self.buffer = MultiAgentReplayBuffer(config['buffer_size'], config)

    def _init_networks(self):
        #初始化actor容器
        self.actors, self.target_actors = {}, {}

        #sbs
        self.actors['sbs'] = {0: SBS_Actor_model(self.gat_out_dim, self.K).to(self.device)}
        self.target_actors['sbs'] = {0: SBS_Actor_model(self.gat_out_dim, self.K).to(self.device)}

        #tbs
        self.actors['tbs'] = {i: TBS_Actor_model(self.gat_out_dim, self.W, self.K).to(self.device) for i in
                              range(self.num_tbs)}
        self.target_actors['tbs'] = {i: TBS_Actor_model(self.gat_out_dim, self.W, self.K).to(self.device) for i in
                                     range(self.num_tbs)}

        #fap
        self.actors['fap'] = {i: FAP_Actor_model(self.gat_out_dim, self.N, self.num_tbs).to(self.device) for i in
                              range(self.num_fap)}
        self.target_actors['fap'] = {i: FAP_Actor_model(self.gat_out_dim, self.N, self.num_tbs).to(self.device) for i in
                                     range(self.num_fap)}

        # 创建临时回放缓冲区，用于复用维度计算逻辑
        temp_buffer = MultiAgentReplayBuffer(1, self.config)

        #critic 网络
        self.critic = SharedCritic_model(temp_buffer.total_obs_dim, temp_buffer.total_action_dim).to(self.device)
        self.target_critic = SharedCritic_model(temp_buffer.total_obs_dim, temp_buffer.total_action_dim).to(self.device)

        # 同步网络参数
        self.target_state_encoder.load_state_dict(self.state_encoder.state_dict())
        for agent_type in self.actors:
            for agent_id in self.actors[agent_type]:
                self.target_actors[agent_type][agent_id].load_state_dict(self.actors[agent_type][agent_id].state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _init_optimizers(self):
        #初始化actor优化器容器
        self.actor_optimizers = {'sbs': {}, 'tbs': {}, 'fap': {}}

        # sbs 优化器
        self.actor_optimizers['sbs'][0] = optim.Adam(self.actors['sbs'][0].parameters(), lr=self.lr_actor)

        # tbs 优化器
        for agent_id, actor in self.actors['tbs'].items():
            self.actor_optimizers['tbs'][agent_id] = optim.Adam(actor.parameters(), lr=self.lr_actor)

        # fap 优化器
        for agent_id, actor in self.actors['fap'].items():
            self.actor_optimizers['fap'][agent_id] = optim.Adam(actor.parameters(), lr=self.lr_actor)

        # gat编码器的优化器
        self.encoder_optimizer = optim.Adam(self.state_encoder.parameters(), lr=self.config['lr_encoder'])

        # critic 优化器
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic, weight_decay=1e-2)

    def _preprocess_observations_batch(self, obs_batch, encoder):
        """处理一个批次的扁平化观察数据，并用GAT进行编码"""
        batch_size = obs_batch.shape[0]
        #拆分批数据
        sbs_obs, tbs_obs, fap_obs = self._split_obs_batch(obs_batch)
        all_obs_nodes = [sbs_obs] + tbs_obs + fap_obs

        #零填充
        padded_obs = torch.zeros(batch_size, self.num_nodes, self.max_obs_dim, device=self.device)
        for i, node_obs_batch in enumerate(all_obs_nodes):
            obs_dim = node_obs_batch.shape[1]
            padded_obs[:, i, :obs_dim] = node_obs_batch

        adj = self.adj

        encoded_states_list = [encoder(padded_obs[i], adj[i]) for i in range(batch_size)]
        encoded_states_batch = torch.stack(encoded_states_list)

        encoded_sbs = encoded_states_batch[:, 0, :]
        encoded_tbs = [encoded_states_batch[:, 1 + i, :] for i in range(self.num_tbs)]
        encoded_fap = [encoded_states_batch[:, 1 + self.num_tbs + i, :] for i in range(self.num_fap)]

        return encoded_sbs, encoded_tbs, encoded_fap

    def select_actions(self, obs):
        actions = {'sbs': {}, 'tbs': {}, 'fap': {}}
        with torch.no_grad():

            obs_batch = self.buffer._flatten_obs(obs).unsqueeze(0)
            encoded_sbs, encoded_tbs, encoded_fap = self._preprocess_observations_batch(obs_batch, self.state_encoder)

            encoded_obs = {'sbs': {0: encoded_sbs[0]},
                           'tbs': {i: encoded_tbs[i][0] for i in range(self.num_tbs)},
                           'fap': {i: encoded_fap[i][0] for i in range(self.num_fap)}}

            # 使用编码后的状态来选择动作
            for agent_type, agent_group in encoded_obs.items():
                for agent_id, agent_encoded_obs in agent_group.items():
                    obs_tensor = agent_encoded_obs.unsqueeze(0)
                    action, _ = self.actors[agent_type][agent_id].get_action(obs_tensor, reparameterize=False)
                    actions[agent_type][agent_id] = action.cpu().squeeze(0)
        return actions

    def _split_obs_batch(self, obs_batch):
        dims, current_idx = self.obs_dims, 0
        sbs_obs = obs_batch[:, current_idx: current_idx + dims['sbs']];
        current_idx += dims['sbs']
        tbs_obs = [obs_batch[:, current_idx + i * dims['tbs']: current_idx + (i + 1) * dims['tbs']] for i in
                   range(self.num_tbs)];
        current_idx += self.num_tbs * dims['tbs']
        fap_obs = [obs_batch[:, current_idx + i * dims['fap']: current_idx + (i + 1) * dims['fap']] for i in
                   range(self.num_fap)]
        return sbs_obs, tbs_obs, fap_obs

    def _soft_update(self):
        for target, local in zip(self.target_critic.parameters(), self.critic.parameters()):
            target.data.copy_(self.tau * local.data + (1.0 - self.tau) * target.data)
        for agent_type in self.actors:
            for agent_id in self.actors[agent_type]:
                for target, local in zip(self.target_actors[agent_type][agent_id].parameters(),
                                         self.actors[agent_type][agent_id].parameters()):
                    target.data.copy_(self.tau * local.data + (1.0 - self.tau) * target.data)

        for target, local in zip(self.target_state_encoder.parameters(), self.state_encoder.parameters()):
            target.data.copy_(self.tau * local.data + (1.0 - self.tau) * target.data)

    def update(self):
        if self.buffer.size < self.batch_size: return
        obs, actions, rewards, future_returns, next_obs, dones = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            sbs_next_obs, tbs_next_obs, fap_next_obs = self._split_obs_batch(next_obs)
            next_actions_list = [self.target_actors['sbs'][0].get_action(sbs_next_obs, reparameterize=False)[0]] + \
                                [self.target_actors['tbs'][i].get_action(tbs_next_obs[i], reparameterize=False)[0] for i
                                 in range(self.num_tbs)] + \
                                [self.target_actors['fap'][i].get_action(fap_next_obs[i])[0] for i in
                                 range(self.num_fap)]
            next_actions = torch.cat(next_actions_list, dim=1)
            q_next = self.target_critic(next_obs, next_actions)
            td_target = rewards + self.gamma * q_next * (1 - dones)
            mc_target = future_returns
            target_q = (1 - self.lambda_target_mix) * td_target + self.lambda_target_mix * mc_target

        current_q = self.critic(obs, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        for optim_group in self.actor_optimizers.values():
            for optimizer in optim_group.values():
                optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()

        encoded_sbs, encoded_tbs, encoded_fap = self._preprocess_observations_batch(obs, self.state_encoder)

        pred_actions_list = [self.actors['sbs'][0].get_action(encoded_sbs, reparameterize=True)[0]] + \
                            [self.actors['tbs'][i].get_action(encoded_tbs[i], reparameterize=True)[0] for i in
                             range(self.num_tbs)] + \
                            [self.actors['fap'][i].get_action(encoded_fap[i])[0] for i in range(self.num_fap)]
        pred_actions = torch.cat(pred_actions_list, dim=1)

        actor_loss = -self.critic(obs, pred_actions).mean()
        actor_loss.backward()

        for optim_group in self.actor_optimizers.values():
            for optimizer in optim_group.values():
                optimizer.step()
        self.encoder_optimizer.step()

        self._soft_update()

    def run_training(self, num_episodes, max_steps_per_episode):
        env = HRCSMultiAgentEnv(self.config)

        for episode in range(num_episodes):
            obs = env.reset()
            episode_instant_reward = 0
            episode_start_ptr = self.buffer.ptr
            for step in range(max_steps_per_episode):

                actions = self.select_actions(obs)
                next_obs, reward, _ = env.step(actions)
                is_episode_end = (step == max_steps_per_episode - 1)

                self.buffer.store(obs, actions, reward, next_obs, is_episode_end)
                self.update()
                obs = next_obs
                episode_instant_reward += reward

            goal_value = float(LPValueProvider.value)
            episode_end_ptr = self.buffer.ptr
            self.buffer.post_process_episode(episode_start_ptr, episode_end_ptr, goal_value)

            if (episode + 1) % 10 == 0:
                print(
                    f"Episode {episode + 1}/{num_episodes}, Goal Value: {goal_value}, Total Instant Reward: {episode_instant_reward:.2f}")


