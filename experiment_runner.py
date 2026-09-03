import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ota_env import OTAEnv
from stable_baselines3 import PPO
from baselines import BaselineRunner

Path("results/final").mkdir(parents=True, exist_ok=True)

N_BLOCKS = 16   # Must match training

def get_valid_action(env, model_obs):
    """Fallback: choose a random unprocessed block"""
    unprocessed = np.where(env.mask == 1)[0]
    if len(unprocessed) == 0:
        return [0, 0]
    block_idx = np.random.choice(unprocessed)
    operation = np.random.randint(0, 3)
    return [block_idx, operation]

def evaluate_model(model, env, num_episodes=20):
    payloads = []
    memories = []
    rewards = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        
        while not done and steps < env.n_blocks + 15:
            try:
                action, _ = model.predict(obs, deterministic=True)
                # Check if action is valid
                block_idx = action[0]
                if env.mask[block_idx] == 0:
                    action = get_valid_action(env, obs)   # fallback
            except:
                action = get_valid_action(env, obs)
            
            obs, reward, done, _, info = env.step(action)
            ep_reward += reward
            steps += 1
        
        payloads.append(info.get('payload_bytes', 0))
        memories.append(info.get('memory_used', 0))
        rewards.append(ep_reward)
        
        print(f"  Episode {ep+1:2d}: Payload = {info.get('payload_bytes', 0):.1f} | Steps = {steps}")
    
    return {
        'mean_payload': float(np.mean(payloads)),
        'std_payload': float(np.std(payloads)),
        'mean_memory': float(np.mean(memories)),
        'mean_reward': float(np.mean(rewards))
    }


if __name__ == "__main__":
    print(" Starting Generic vs BD Experiment (with fallback)...\n")
    
    # Load model
    try:
        model_generic = PPO.load("results/models/generic_best/best_model.zip")
        print(" Loaded best Generic model")
    except:
        model_generic = PPO.load("results/models/ppo_generic_final")
        print(" Loaded final Generic model")

    # Evaluations
    print(f"Evaluating with n_blocks = {N_BLOCKS} ...")
    
    env_generic = OTAEnv(n_blocks=N_BLOCKS, constrained_network_mode=False)
    generic_rl = evaluate_model(model_generic, env_generic, num_episodes=15)   # Reduced episodes for speed
    
    env_constrained = OTAEnv(n_blocks=N_BLOCKS, constrained_network_mode=True)
    bd_rl = evaluate_model(model_generic, env_constrained, num_episodes=15)
    
    # Baseline
    baseline_runner = BaselineRunner(n_blocks=N_BLOCKS)
    random_baseline = baseline_runner.run_random_baseline(10)
    
    # ====================== RESULTS ======================
    results = {
        'Setting': ['Generic RL (PPO)', 'BD RL (PPO)', 'Random Baseline'],
        'Mean Payload': [generic_rl['mean_payload'], bd_rl['mean_payload'], random_baseline['mean_payload']],
        'Std Payload': [generic_rl['std_payload'], bd_rl['std_payload'], random_baseline['std_payload']],
        'Mean Memory': [generic_rl['mean_memory'], bd_rl['mean_memory'], random_baseline['mean_memory']],
        'Mean Reward': [generic_rl['mean_reward'], bd_rl['mean_reward'], 'N/A']
    }
    
    df = pd.DataFrame(results)
    print("\n" + "="*75)
    print("FINAL RESULTS: Generic vs Bangladesh Conditions")
    print("="*75)
    print(df.round(2))
    
    df.to_csv("results/final/results_comparison.csv", index=False)
    print("\n Results saved!")

    # Plot
    plt.figure(figsize=(14, 6))
    settings = ['Generic RL', 'BD RL', 'Random']
    plt.subplot(1, 2, 1)
    plt.bar(settings, [generic_rl['mean_payload'], bd_rl['mean_payload'], random_baseline['mean_payload']], 
            color=['#1f77b4', '#ff7f0e', '#7f7f7f'])
    plt.ylabel('Mean Payload Cost')
    plt.title('Payload Size Comparison')
    plt.grid(axis='y', alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.bar(settings, [generic_rl['mean_memory'], bd_rl['mean_memory'], random_baseline['mean_memory']], 
            color=['#1f77b4', '#ff7f0e', '#7f7f7f'])
    plt.ylabel('Mean Memory Overhead')
    plt.title('Memory Usage Comparison')
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/final/final_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(" Plot saved as results/final/final_comparison.png")
    print("\n Checkpoint 5 Completed!")