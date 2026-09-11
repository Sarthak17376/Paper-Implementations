# paste train.py contents here

"""
Entry point: trains our own GATv2 Actor-Critic + PPO implementation
against the real TransitEnv (rl/env.py) from the AlphaTransit repo,
using the actual Bloomington dataset and UXsim traffic simulator.

Features:
- Robust checkpointing (saves model, optimizer, RNG states, history)
- Resume capability (--resume) to chain multiple Kaggle sessions
- Time guard (--max_hours) to exit cleanly before Kaggle's 12-hour session kill
- Offline-first local data loading (--data_source csv) to eliminate HF Hub warnings
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from env import TransitEnv  # our own from-scratch environment
from models import ModelConfig
from ppo_agent import PPOAgent, PPOConfig, RolloutBuffer, Transition, obs_to_valid_mask


def parse_args():
    p = argparse.ArgumentParser()
    # Data arguments (defaults to local offline files to avoid HF Hub warnings/rate limits)
    p.add_argument("--data_source", type=str, choices=["csv", "huggingface"], default="csv")
    p.add_argument("--nodes_csv", type=str, default="./data/bloomington_nodes.csv")
    p.add_argument("--links_csv", type=str, default="./data/bloomington_links.csv")
    p.add_argument("--demand_csv", type=str, default="./data/bloomington_demand.csv")
    p.add_argument("--routes_json", type=str, default="./data/bloomington_routes.json")
    p.add_argument("--hf_dataset_name", type=str, default="matrix-multiply/bloomington-tndp")
    p.add_argument("--hf_split", type=str, default="benchmark")

    # Environment & RL hyperparameters
    p.add_argument("--num_routes", type=int, default=16)
    p.add_argument("--max_route_length", type=int, default=14)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--t_sim", type=float, default=10000.0)
    p.add_argument("--num_episodes", type=int, default=200)
    p.add_argument("--rollout_steps", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Checkpointing & Kaggle quota / session management
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    p.add_argument("--save_every", type=int, default=5, help="Save periodic checkpoint every N episodes")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt file to resume from")
    p.add_argument(
        "--max_hours",
        type=float,
        default=11.0,
        help="Max hours to run before graceful exit (Kaggle hard-kills at 12 hours)",
    )
    return p.parse_args()


def build_env(args):
    common_kwargs = dict(
        num_routes=args.num_routes,
        max_route_length=args.max_route_length,
        alpha=args.alpha,
        t_sim=args.t_sim,
        seed=args.seed,
    )
    if args.data_source == "huggingface":
        from datasets import load_dataset
        nodes = load_dataset(args.hf_dataset_name, "nodes", split=args.hf_split)
        links = load_dataset(args.hf_dataset_name, "links", split=args.hf_split)
        demand = load_dataset(args.hf_dataset_name, "demand", split=args.hf_split)
        routes = load_dataset(
            "json",
            data_files=f"hf://datasets/{args.hf_dataset_name}/standard/bloomington_existing_routes.json",
            split="train",
        )
        return TransitEnv.from_huggingface(nodes, links, demand, routes, **common_kwargs)
    else:
        assert args.nodes_csv and args.links_csv and args.demand_csv and args.routes_json, (
            "--nodes_csv/--links_csv/--demand_csv/--routes_json are required when --data_source=csv"
        )
        return TransitEnv.from_csv(args.nodes_csv, args.links_csv, args.demand_csv, args.routes_json, **common_kwargs)


def run_episode(env: TransitEnv, agent: PPOAgent, buffer: RolloutBuffer, max_steps: int):
    obs, info = env.reset()
    episode_reward = 0.0
    ep_done = False
    steps = 0

    while not ep_done and steps < max_steps:
        n_nodes = env.n_nodes
        valid_mask_np = obs_to_valid_mask(obs, n_nodes)

        action, log_prob, value, _ = agent.act(obs)
        next_obs, reward, route_done, ep_done_flag, info = env.step(action)
        ep_done = bool(ep_done_flag)

        buffer.add(
            Transition(
                node_features=obs["node_features"].copy(),
                edge_index=obs["edge_index"].copy(),
                edge_features=obs["edge_features"].copy(),
                route_progress=obs["route_progress"].copy(),
                frontier_index=int(obs["frontier_index"]),
                valid_mask=valid_mask_np.copy(),
                action=action,
                log_prob=log_prob,
                value=value,
                reward=reward,
                done=ep_done,
            )
        )
        episode_reward += reward
        obs = next_obs
        steps += 1

    return episode_reward, steps, obs


def save_checkpoint(path: str, episode: int, agent: PPOAgent, ep_reward: float, best_reward: float, history: list, args):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rng_state = {
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }
    checkpoint = {
        "episode": episode,
        "model_state_dict": agent.model.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "rng_state": rng_state,
        "last_reward": ep_reward,
        "best_reward": best_reward,
        "history": history,
        "args": vars(args),
    }
    torch.save(checkpoint, path)

    # Save human-readable JSON history log alongside checkpoints
    log_path = os.path.join(os.path.dirname(os.path.abspath(path)), "training_history.json")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    env = build_env(args)
    print(
        f"TransitEnv ready: n_nodes={env.n_nodes}, num_routes={env.num_routes}, "
        f"max_route_length={env.max_route_length}"
    )

    model_cfg = ModelConfig(
        node_feat_dim=16,
        edge_feat_dim=2,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_routes=env.num_routes,
    )
    ppo_cfg = PPOConfig(lr=args.lr, gamma=args.gamma, rollout_steps=args.rollout_steps, device=args.device)
    agent = PPOAgent(model_cfg, ppo_cfg)
    print(f"Using device: {ppo_cfg.device}")

    # Checkpoint & Resume initialization
    start_episode = 1
    best_reward = -float("inf")
    history = []

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=agent.device, weights_only=False)
            agent.model.load_state_dict(checkpoint["model_state_dict"])
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            rng = checkpoint.get("rng_state", {})
            if "torch" in rng and rng["torch"] is not None:
                torch.set_rng_state(rng["torch"].cpu())
            if "torch_cuda" in rng and rng["torch_cuda"] is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all([s.cpu() for s in rng["torch_cuda"]])
                except Exception:
                    pass
            if "numpy" in rng:
                np.random.set_state(rng["numpy"])
            if "random" in rng:
                random.setstate(rng["random"])

            start_episode = checkpoint["episode"] + 1
            best_reward = checkpoint.get("best_reward", -float("inf"))
            history = checkpoint.get("history", [])
            print(
                f"--> Successfully resumed from episode {checkpoint['episode']}. "
                f"Next episode: {start_episode}. Best reward so far: {best_reward:.3f}"
            )
        else:
            print(f"--> [Warning] Checkpoint '{args.resume}' not found. Starting from episode 1.")

    buffer = RolloutBuffer()
    max_steps_per_episode = env.num_routes * env.max_route_length + env.num_routes  # generous cap

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    start_wall_time = time.time()
    max_seconds = args.max_hours * 3600.0 if args.max_hours > 0 else float("inf")

    last_ep_reward = 0.0

    for episode in range(start_episode, args.num_episodes + 1):
        # Time guard: exit cleanly before Kaggle's 12-hour session kill
        elapsed_total = time.time() - start_wall_time
        if elapsed_total >= max_seconds:
            print(f"\n[Time Guard] Elapsed time {elapsed_total / 3600.0:.2f}h reached safety limit ({args.max_hours:.2f}h).")
            print(f"[Time Guard] Saving checkpoint at completed episode {episode - 1} and exiting cleanly.")
            save_checkpoint(
                os.path.join(args.checkpoint_dir, "checkpoint_latest.pt"),
                episode - 1,
                agent,
                last_ep_reward,
                best_reward,
                history,
                args,
            )
            break

        t0 = time.time()
        buffer.clear()
        ep_reward, steps, last_obs = run_episode(env, agent, buffer, max_steps_per_episode)
        last_ep_reward = ep_reward

        # Bootstrap value for the final state (0 if truly terminal)
        with torch.no_grad():
            if steps >= max_steps_per_episode:
                _, _, last_value, _ = agent.act(last_obs)
            else:
                last_value = 0.0

        stats = agent.update(buffer, last_value)
        dt = time.time() - t0

        record = {
            "episode": episode,
            "steps": steps,
            "reward": float(ep_reward),
            "policy_loss": float(stats["policy_loss"]),
            "value_loss": float(stats["value_loss"]),
            "entropy": float(stats["entropy"]),
            "duration_s": float(dt),
        }
        history.append(record)

        if episode % args.log_every == 0:
            print(
                f"episode {episode:4d}/{args.num_episodes} | steps {steps:3d} | reward {ep_reward:8.3f} | "
                f"policy_loss {stats['policy_loss']:.4f} | value_loss {stats['value_loss']:.4f} | "
                f"entropy {stats['entropy']:.4f} | {dt:.1f}s"
            )

        # 1. Always save latest checkpoint (guards against mid-run crashes)
        save_checkpoint(
            os.path.join(args.checkpoint_dir, "checkpoint_latest.pt"),
            episode,
            agent,
            ep_reward,
            best_reward,
            history,
            args,
        )

        # 2. Save periodic checkpoint
        if episode % args.save_every == 0:
            save_checkpoint(
                os.path.join(args.checkpoint_dir, f"checkpoint_ep_{episode}.pt"),
                episode,
                agent,
                ep_reward,
                best_reward,
                history,
                args,
            )
            print(f"--> Saved periodic checkpoint: checkpoint_ep_{episode}.pt")

        # 3. Save best checkpoint
        if ep_reward > best_reward:
            best_reward = ep_reward
            save_checkpoint(
                os.path.join(args.checkpoint_dir, "checkpoint_best.pt"),
                episode,
                agent,
                ep_reward,
                best_reward,
                history,
                args,
            )
            print(f"--> New best reward: {best_reward:.3f}! Saved checkpoint_best.pt")

    print(f"\nTraining run complete. Checkpoints and training history saved to '{args.checkpoint_dir}'.")


if __name__ == "__main__":
    main()
