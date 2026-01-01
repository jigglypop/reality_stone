import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class SimplePolicy(nn.Module):
    """
    간단한 MLP 정책 네트워크.
    다유닛 격자 환경에서 유닛 관측 -> 행동 확률
    행동: 이동(유닛 타입별 방향 수) + 공격(1)
    최대 9개 action (8방향 이동 + 공격)
    """
    def __init__(self, input_dim=12, hidden_dim=64, output_dim=9):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # 교대 턴(체스 룰)에서 "어느 유닛이 행동할지" 선택을 위한 head
        self.fc_sel = nn.Linear(hidden_dim, 1)
        
    def forward_logits(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h)

    def forward_select_logit(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_sel(h).squeeze(-1)

    def forward(self, x):
        logits = self.forward_logits(x)
        return F.softmax(logits, dim=-1)

    def get_action(self, obs, action_mask=None):
        logits = self.forward_logits(obs)

        if action_mask is not None:
            mask = action_mask.to(dtype=torch.bool)
            masked_logits = logits.clone()
            masked_logits[~mask] = -1e9
            logits = masked_logits

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

def get_turn_action_with_env(policy: SimplePolicy, env, f: int, unit_obs: List[torch.Tensor]) -> Tuple[int, int, torch.Tensor]:
    """
    GridCombatEnv는 유닛별 이동 패턴 길이(len(move_pattern))에 따라 action 의미가 달라집니다.
    따라서 "유닛 타입별 고정 action 공간"이 아니라, env의 패턴 길이에 맞춘 마스킹을 사용해야 합니다.

    반환: (unit_index, action, log_prob_total)
    """
    if len(unit_obs) == 0:
        return 0, 0, torch.tensor(0.0)

    alive_mask: List[bool] = []
    for u in unit_obs:
        if isinstance(u, torch.Tensor) and u.numel() > 2:
            alive_mask.append(float(u[2].item()) > 0.0)
        else:
            alive_mask.append(False)

    if not any(alive_mask):
        return 0, 0, torch.tensor(0.0)

    sel_logits = []
    for ok, u in zip(alive_mask, unit_obs):
        if not ok:
            sel_logits.append(u.new_tensor(-1e9) if isinstance(u, torch.Tensor) else torch.tensor(-1e9))
            continue
        sel_logits.append(policy.forward_select_logit(u))
    sel_logits = torch.stack(sel_logits)
    sel_probs = F.softmax(sel_logits, dim=-1)
    sel_dist = torch.distributions.Categorical(sel_probs)
    ui = int(sel_dist.sample().item())
    logp_sel = sel_dist.log_prob(torch.tensor(ui))

    uobs = unit_obs[ui]
    in_range = float(uobs[11].item()) if (isinstance(uobs, torch.Tensor) and uobs.numel() > 11) else 0.0

    # env 기반 이동 패턴 길이로 action 공간을 정의한다.
    t = int(env.unit_types[f][ui])
    pattern_id = int(env.typespecs[f][t].get("pattern", 0))
    move_pattern, _, _ = env.MOVE_PATTERN_POOL[int(pattern_id)]
    move_dirs = int(len(move_pattern))
    attack_action = int(move_dirs)  # env.step에서 action >= len(move_pattern)면 공격

    # policy 출력 dim(9) 내에 있어야 함: 최대 패턴 길이가 8이므로 attack_action은 최대 8
    move_dirs = max(1, min(8, move_dirs))
    attack_action = max(1, min(8, attack_action))

    mask = torch.zeros(9, dtype=torch.bool)
    for i in range(move_dirs):
        mask[i] = True
    mask[attack_action] = in_range > 0.0

    # 사거리 안이면 공격 강제(퇴화: 이동만 반복 방지)
    if in_range > 0.0:
        for i in range(move_dirs):
            mask[i] = False
        mask[attack_action] = True

    a, logp_a = policy.get_action(uobs, action_mask=mask)
    return ui, a, (logp_sel + logp_a)


def train_one_episode(env, policies: List[SimplePolicy], optimizers: List[torch.optim.Optimizer], first_turn=None):
    """
    한 에피소드 진행 및 REINFORCE 업데이트 (공유 정책, faction 단위 보상)
    first_turn: None이면 랜덤, 0 또는 1이면 해당 진영 선턴
    """
    obs = env.reset(first_turn=first_turn)
    done = False
    
    # timestep마다 (턴에서의 logprob) 저장
    log_probs_sum = [[], []]
    rewards_episode = [[], []]  # faction reward
    
    while not done:
        obs0_units, obs1_units = obs
        side = int(getattr(env, "side_to_act", 0))
        if side == 0:
            ui, a, lp = get_turn_action_with_env(policies[0], env, 0, obs0_units)
            next_obs, rewards, done, info = env.step((0, ui, a))
            log_probs_sum[0].append(lp)
            log_probs_sum[1].append(torch.tensor(0.0))
        else:
            ui, a, lp = get_turn_action_with_env(policies[1], env, 1, obs1_units)
            next_obs, rewards, done, info = env.step((1, ui, a))
            log_probs_sum[0].append(torch.tensor(0.0))
            log_probs_sum[1].append(lp)
        rewards_episode[0].append(float(rewards[0]))
        rewards_episode[1].append(float(rewards[1]))

        obs = next_obs

    # 학습 (REINFORCE)
    for pid in range(2):
        R = 0
        loss = 0
        returns = []
        # Return 계산
        for r in reversed(rewards_episode[pid]):
            R = r + 0.95 * R # gamma = 0.95
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        if returns.numel() > 1:
            std = returns.std(unbiased=False)
            if std > 1e-8:
                returns = (returns - returns.mean()) / (std + 1e-8)
            
        for lp, ret in zip(log_probs_sum[pid], returns):
            loss -= lp * ret
            
        optimizers[pid].zero_grad()
        # 에피소드가 첫 턴에 끝나는 등, 해당 pid가 실제로 한 행동이 없으면
        # loss가 파라미터 그래프에 연결되지 않을 수 있습니다.
        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            loss.backward()
            optimizers[pid].step()
        
    return info

