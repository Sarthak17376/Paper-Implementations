"""
visualize.py - Publication-grade & interactive visualizations for the
Bloomington Transit Route Network Design Problem (TRNDP).
Replicates Figures 1 & 2 from Poudel & Li (arXiv:2512.19767).
"""
from __future__ import annotations

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRANSIT_CENTER_NODE = "96"  # Highlighted in paper Figure 2


def load_network_for_vis(
    nodes_path: str = "./data/bloomington_nodes.csv",
    links_path: str = "./data/bloomington_links.csv",
    demand_path: str = "./data/bloomington_demand.csv",
    routes_path: str = "./data/bloomington_routes.json",
):
    nodes_df = pd.read_csv(nodes_path)
    links_df = pd.read_csv(links_path)
    demand_df = pd.read_csv(demand_path) if os.path.exists(demand_path) else None

    with open(routes_path, "r", encoding="utf-8") as f:
        routes_raw = json.load(f)

    routes_dict = {
        r["name"]: [str(n) for n in r["nodes"]]
        for r in routes_raw
    }
    node_xy = {str(row["name"]): (float(row["x"]), float(row["y"])) for _, row in nodes_df.iterrows()}
    return nodes_df, links_df, demand_df, routes_dict, node_xy


def plot_network_routes(
    routes_dict: dict[str, list[str]],
    node_xy: dict[str, tuple[float, float]],
    links_df: pd.DataFrame,
    title: str = "Transit Route Network",
    ax: plt.Axes | None = None,
    show_legend: bool = True,
    transit_center_id: str = TRANSIT_CENTER_NODE,
) -> plt.Axes:
    """Plots road network graph in light gray and overlaid transit routes in distinct colors."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9), dpi=150)

    # 1. Base street network (Definition 1)
    for _, link in links_df.iterrows():
        u, v = str(link["start"]), str(link["end"])
        if u in node_xy and v in node_xy:
            x0, y0 = node_xy[u]
            x1, y1 = node_xy[v]
            ax.plot([x0, x1], [y0, y1], color="#d0d0d0", linewidth=1.1, zorder=1)

    # 2. Base nodes
    xs = [node_xy[nid][0] for nid in node_xy]
    ys = [node_xy[nid][1] for nid in node_xy]
    ax.scatter(xs, ys, color="#aaaaaa", s=14, zorder=2, alpha=0.6)

    # 3. Palette for routes (using tab20 for up to 20 distinct high-contrast colors)
    color_map = plt.cm.tab20(np.linspace(0, 1, max(len(routes_dict), 1)))

    for (route_name, nodes), color in zip(routes_dict.items(), color_map):
        valid_nodes = [str(n) for n in nodes if str(n) in node_xy]
        if len(valid_nodes) < 2:
            continue
        rx = [node_xy[n][0] for n in valid_nodes]
        ry = [node_xy[n][1] for n in valid_nodes]
        ax.plot(rx, ry, color=color, linewidth=2.4, alpha=0.85, label=route_name, zorder=3)
        ax.scatter(rx, ry, color=color, s=22, zorder=4)

    # 4. Highlight Transit Center (Node 96) with a prominent marker (Figure 2)
    if transit_center_id in node_xy:
        tc_x, tc_y = node_xy[transit_center_id]
        ax.scatter(
            [tc_x], [tc_y],
            color="red",
            marker="*",
            s=280,
            edgecolor="black",
            linewidth=1.2,
            zorder=6,
            label=f"Transit Center ({transit_center_id})",
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_aspect("equal")
    ax.axis("off")
    if show_legend:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8, ncol=1)
    return ax


def plot_paper_comparison(
    real_routes: dict[str, list[str]],
    rl_routes: dict[str, list[str]],
    node_xy: dict[str, tuple[float, float]],
    links_df: pd.DataFrame,
    save_path: str = "figure2_comparison.png",
):
    """Replicates Figure 2 from Poudel & Li: Real-world vs. RL-designed routes side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=200)

    plot_network_routes(real_routes, node_xy, links_df, title="Real-world Baseline (Bloomington Transit)", ax=axes[0], show_legend=False)
    plot_network_routes(rl_routes, node_xy, links_df, title="RL-Designed Routes (GATv2 + PPO)", ax=axes[1], show_legend=False)

    fig.suptitle("Comparison of Real-World and RL-Designed Transit Route Networks\n(Bloomington, Indiana — 143 Nodes, 243 Edges)", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved side-by-side comparison figure to: {save_path}")
    plt.show()


def create_interactive_route_map(
    routes_dict: dict[str, list[str]],
    node_xy: dict[str, tuple[float, float]],
    links_df: pd.DataFrame,
    save_html_path: str = "transit_routes_interactive.html",
):
    """Creates an interactive Plotly map where routes can be toggled on/off in the browser."""
    import plotly.graph_objects as go

    fig = go.Figure()

    # 1. Add background street network edges
    edge_x, edge_y = [], []
    for _, link in links_df.iterrows():
        u, v = str(link["start"]), str(link["end"])
        if u in node_xy and v in node_xy:
            edge_x += [node_xy[u][0], node_xy[v][0], None]
            edge_y += [node_xy[u][1], node_xy[v][1], None]

    fig.add_trace(
        go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.0, color="#d0d0d0"),
            hoverinfo="none",
            mode="lines",
            name="Road Network",
            showlegend=True,
        )
    )

    # 2. Add base nodes
    node_x = [node_xy[nid][0] for nid in node_xy]
    node_y = [node_xy[nid][1] for nid in node_xy]
    node_text = [f"Node: {nid}" for nid in node_xy]

    fig.add_trace(
        go.Scatter(
            x=node_x, y=node_y,
            mode="markers",
            marker=dict(size=4, color="#999999"),
            text=node_text,
            hoverinfo="text",
            name="Intersections / Stops",
        )
    )

    # 3. Add each transit route as a toggleable trace
    for route_name, nodes in routes_dict.items():
        valid_nodes = [str(n) for n in nodes if str(n) in node_xy]
        if len(valid_nodes) < 2:
            continue
        rx = [node_xy[n][0] for n in valid_nodes]
        ry = [node_xy[n][1] for n in valid_nodes]
        hover_labels = [f"{route_name}<br>Stop {i+1}: Node {n}" for i, n in enumerate(valid_nodes)]

        fig.add_trace(
            go.Scatter(
                x=rx, y=ry,
                mode="lines+markers",
                name=route_name,
                line=dict(width=3.5),
                marker=dict(size=7),
                text=hover_labels,
                hoverinfo="text",
            )
        )

    # 4. Highlight Transit Center (Node 96)
    if TRANSIT_CENTER_NODE in node_xy:
        tc_x, tc_y = node_xy[TRANSIT_CENTER_NODE]
        fig.add_trace(
            go.Scatter(
                x=[tc_x], y=[tc_y],
                mode="markers+text",
                marker=dict(size=14, color="red", symbol="star"),
                name="Transit Center (96)",
                text=["Transit Center (Node 96)"],
                textposition="top center",
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title="Interactive Bloomington Transit Route Network (Click Legend to Toggle Routes)",
        template="plotly_white",
        showlegend=True,
        legend=dict(title="Toggle Routes:", yanchor="top", y=0.99, xanchor="left", x=1.02),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        width=950,
        height=750,
    )

    if save_html_path:
        fig.write_html(save_html_path)
        print(f"Interactive route map saved to: {save_html_path}")
    return fig


def plot_training_metrics(
    history_path: str = "./checkpoints/training_history.json",
    save_path: str = "training_metrics.png",
):
    """Plots episode rewards, policy loss, value loss, and entropy from training_history.json."""
    if not os.path.exists(history_path):
        print(f"No history file found at {history_path}. Run training first.")
        return

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    if not history:
        print("History is empty.")
        return

    df = pd.DataFrame(history)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)

    # Reward
    axes[0, 0].plot(df["episode"], df["reward"], color="#1f77b4", linewidth=1.5)
    axes[0, 0].set_title("Episode Reward (Psi, Omega, Travel Time)", fontweight="bold")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)

    # Policy & Value Loss
    axes[0, 1].plot(df["episode"], df["policy_loss"], color="#ff7f0e", label="Policy Loss")
    axes[0, 1].plot(df["episode"], df["value_loss"], color="#2ca02c", label="Value Loss")
    axes[0, 1].set_title("Loss Trajectories", fontweight="bold")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)

    # Policy Entropy
    axes[1, 0].plot(df["episode"], df["entropy"], color="#d62728", linewidth=1.5)
    axes[1, 0].set_title("Policy Entropy (Exploration)", fontweight="bold")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Entropy")
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)

    # Duration per episode
    axes[1, 1].plot(df["episode"], df["duration_s"], color="#9467bd", linewidth=1.5)
    axes[1, 1].set_title("Computation Time per Episode (seconds)", fontweight="bold")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Seconds")
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("PPO Training Progress & Convergence", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Training metrics plot saved to: {save_path}")
    plt.show()


def generate_routes_from_checkpoint(
    checkpoint_path: str = "./checkpoints/checkpoint_latest.pt",
    data_dir: str = "./data",
    num_routes: int = 16,
    max_route_length: int = 14,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, list[str]]:
    """
    Loads policy weights from a saved checkpoint, runs a fast rollout to build
    all routes, and returns the designed route dictionary without waiting
    for a full traffic simulation.
    """
    import torch
    from network_data import load_network_data
    from env import TransitEnv
    from models import ModelConfig
    from ppo_agent import PPOAgent, PPOConfig

    net = load_network_data(
        os.path.join(data_dir, "bloomington_nodes.csv"),
        os.path.join(data_dir, "bloomington_links.csv"),
        os.path.join(data_dir, "bloomington_demand.csv"),
        os.path.join(data_dir, "bloomington_routes.json"),
    )
    # Minimal t_sim (1s) since we only need the agent's route construction trajectory, not simulation metrics
    env = TransitEnv(net, num_routes=num_routes, max_route_length=max_route_length, t_sim=1.0, seed=seed)

    model_cfg = ModelConfig(
        node_feat_dim=16,
        edge_feat_dim=2,
        hidden_dim=128,
        num_heads=4,
        num_layers=4,
        num_routes=num_routes,
    )
    ppo_cfg = PPOConfig(device=device)
    agent = PPOAgent(model_cfg, ppo_cfg)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.model.load_state_dict(checkpoint["model_state_dict"])
    agent.model.eval()

    obs, _ = env.reset(seed=seed)
    ep_done = False
    while not ep_done:
        with torch.no_grad():
            action, _, _, _ = agent.act(obs)
        obs, _, _, ep_done, _ = env.step(action)

    return env.completed_routes
