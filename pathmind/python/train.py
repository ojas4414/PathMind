import sys
import torch
sys.path.insert(0,'build/Release')
import traffic_env
import random
from torch.distributions import Categorical
import sys
sys.path.insert(0, 'python')
from agent import PolicyNetwork
policy=PolicyNetwork()
import torch.optim as optim
env = traffic_env.trafficEnv(200)
obs = env.reset()



def collect_episode():
    obs= env.reset()
    states,actions,rewards, log_probs = [], [], [], []
    done=False

    while not done:
        state = torch.tensor(obs, dtype=torch.float32)
        probs = policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        obs, reward, done = env.step(action.item())
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
    return states, actions, rewards, log_probs


GAMMA=0.99
def compute_returns(rewards):
    returns=[]
    R=0
    for r in reversed(rewards):
        R= r + GAMMA*R
        returns.insert(0,R)
    return torch.tensor(returns, dtype=torch.float32)




CLIP = 0.2
EPOCHS = 4
LR = 0.001
optimizer = optim.Adam(policy.parameters(), lr=LR)

def update(states, actions, log_probs_old, returns):
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    for _ in range(EPOCHS):
        states_t = torch.stack(states)
        probs = policy(states_t)
        dist = Categorical(probs)
        log_probs_new = dist.log_prob(torch.stack(actions))
        
        ratio = torch.exp(log_probs_new - torch.stack(log_probs_old).detach())
        clipped = torch.clamp(ratio, 1 - CLIP, 1 + CLIP)
        loss = -torch.min(ratio * returns, clipped * returns).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


EPISODES = 500

for episode in range(EPISODES):
    states, actions, rewards, log_probs = collect_episode()
    returns = compute_returns(rewards)
    update(states, actions, log_probs, returns)
    
    if episode % 50 == 0:
        print(f"Episode {episode}, Total Reward: {sum(rewards):.1f}")
torch.save(policy.state_dict(), 'trained_policy.pth')
print("Model saved.")










