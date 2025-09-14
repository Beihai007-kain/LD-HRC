import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

class GraphAttentionLayer_model(nn.Module):

    #基础模块，实现图注意力机制，对节点特征进行加权求和
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer_model, self).__init__()

        #防止在向前传播的过程中过拟合
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))

        #最开始可以用Xvaier均匀分布初始化
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))

        #初始化用Xavier均匀分布
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        #计算节点之间的注意力
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        #先做一次线性变换
        Wh = torch.matmul(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        #对每个节点的注意力分数进行归一化
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    #为计算注意力分数准备输入，返回的矩阵将包含每对节点的特征组合。
    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)


class GATStateEncoder_model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, n_heads, dropout, alpha=0.2):
        self.n_heads = n_heads
        self.dropout = dropout
        super(GATStateEncoder_model, self).__init__()
        self.attentions = [GraphAttentionLayer_model(in_features, hidden_features, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer_model(hidden_features * n_heads, out_features, dropout=dropout, alpha=alpha,
                                           concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, 0.1, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, 0.1, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x

class SBS_Actor_model(nn.Module):
    def __init__(self, gat_out_dim, K, hidden_size=256):
        super(SBS_Actor, self).__init__()
        self.K = K
        self.fc1 = nn.Linear(gat_out_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mu_std_out = nn.Linear(hidden_size, K * 2)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mu_std = self.mu_std_out(x)
        mu, log_std = torch.chunk(mu_std, 2, dim=-1)
        mu = torch.tanh(mu)
        std = F.softplus(log_std) + 1e-5
        return mu, std

    def get_action(self, obs, reparameterize=True):
        mu, std = self.forward(obs)
        dist = Normal(mu, std)
        # 这里使用重参数化技巧
        action = dist.rsample() if reparameterize else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob


class TBS_Actor_model(nn.Module):
    def __init__(self, gat_out_dim, W, K, hidden_size=256):
        super(TBS_Actor_model, self).__init__()
        self.W = W
        self.K = K
        self.fc1 = nn.Linear(gat_out_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        #这个地方需要多头策略网络
        self.assignment_head = nn.Linear(hidden_size, W * K)
        self.power_head = nn.Linear(hidden_size, W * 2)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        assignment_logits = self.assignment_head(x).view(-1, self.W, self.K)
        power_mu_std = self.power_head(x)
        power_mu, power_log_std = torch.chunk(power_mu_std, 2, dim=-1)
        power_mu = torch.tanh(power_mu)
        power_std = F.softplus(power_log_std) + 1e-5
        return assignment_logits, power_mu, power_std

    def get_action(self, obs, reparameterize=True):
        assignment_logits, power_mu, power_std = self.forward(obs)
        assignment_dist = Categorical(logits=assignment_logits)
        assignment_action = assignment_dist.sample()
        #创建Normal分布
        power_dist = Normal(power_mu, power_std)
        #这里也使用重参数化技巧
        power_action = power_dist.rsample() if reparameterize else power_dist.sample()
        action = torch.cat([assignment_action.float(), power_action], dim=-1)
        log_prob = assignment_dist.log_prob(assignment_action).sum(dim=-1, keepdim=True) + \
                   power_dist.log_prob(power_action).sum(dim=-1, keepdim=True)
        return action, log_prob


class FAP_Actor_model(nn.Module):
    def __init__(self, gat_out_dim, N, I, hidden_size=256):
        super(FAP_Actor_model, self).__init__()
        self.N = N
        self.num_choices = I + 2
        self.fc1 = nn.Linear(gat_out_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.selection_head = nn.Linear(hidden_size, self.N * self.num_choices)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.selection_head(x).view(-1, self.N, self.num_choices)

    def get_action(self, obs):
        selection_logits = self.forward(obs)
        dist = Categorical(logits=selection_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action.float(), log_prob

class SharedCritic_model(nn.Module):
    def __init__(self, total_obs_dim, total_action_dim, hidden_size=256):
        super(SharedCritic_model, self).__init__()
        input_dim = total_obs_dim + total_action_dim
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.q_out = nn.Linear(hidden_size, 1)

    def forward(self, all_obs, all_actions):

        #需要拼接state和action信息计算对应Q值
        x = torch.cat([all_obs, all_actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.q_out(x)