import math
import torch
import numpy as np

class MCTSNode:
    def __init__(self, state):
        self.state = state
        self.visit_count = 0
        self.total_value = 0.0
        self.children = {}
        self.prior_prob = 0.0  # Probability assigned at parent's expansion time
        self.net_value_est = 0.0  # Approx value from net
        self.net_done_prob = 0.0  # Probability that the puzzle is done
        self.debug_eventual_solution_found = False

def ucb_score(parent, child, c_puct):
    if child.visit_count == 0:
        return float("inf")
    q_value = child.total_value / child.visit_count
    u_value = (
        c_puct
        * child.prior_prob
        * math.sqrt(parent.visit_count)
        / (1 + child.visit_count)
    )
    return q_value + u_value

def select_child(node, c_puct):
    best_action = None
    best_child = None
    best_score = -1e9

    for action, child in node.children.items():
        score = ucb_score(node, child, c_puct=c_puct)
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    return best_action, best_child

def expand_node(
    node,
    row_logits,
    col_logits,
    width_logits,
    height_logits,
    color_logits,
    net_value_est,
    net_done_logits,
    expansions_per_node,
    num_candidates,
    max_attempts_factor=5,
):
    """
    node:        The node to expand
    row_logits:  shape [1, max_rows]
    col_logits:  shape [1, max_cols]
    width_logits: shape [1, max_cols]
    height_logits: shape [1, max_rows]
    color_logits: shape [1, num_colors]
    net_value_est: shape [1, 1]
    net_done_logits: shape [1, 2]
    expansions_per_node: how many child actions we sample at each node
    num_candidates: how many candidates we consider sampling at each node
    """

    # The network is trained to estimate the future reward from this node onwards
    # (excluding the current node)
    node.net_value_est = net_value_est.item()
    done_probs = torch.softmax(net_done_logits, dim=-1)[0]
    # Interpret done_probs[0] = p(notDone), done_probs[1] = p(done).
    node.net_done_prob = done_probs[1].item()

    row_log = row_logits[0].cpu()  # shape [net.max_rows]
    col_log = col_logits[0].cpu()  # shape [net.max_cols]
    width_log = width_logits[0].cpu()  # shape [net.max_cols for widths]
    height_log = height_logits[0].cpu()  # shape [net.max_rows for heights]
    color_log = color_logits[0].cpu()  # shape [num_colors]

    row_dim = node.state.current_guess.shape[0]
    col_dim = node.state.current_guess.shape[1]
    num_colors = node.state.num_colors

    candidates = []
    tries = 0
    max_tries = max_attempts_factor * expansions_per_node
    created_children = 0

    while created_children < num_candidates and tries < max_tries:
        r = np.random.randint(0, row_dim)
        c = np.random.randint(0, col_dim)

        max_w = col_dim - c
        w = np.random.randint(0, max_w)
        max_h = row_dim - r
        h = np.random.randint(0, max_h)

        color = np.random.randint(0, num_colors - 1)

        sum_logits = (
            row_log[r]
            + col_log[c]
            + width_log[w]
            + height_log[h]
            + color_log[color]
        )
        action_tuple = (r, c, w, h, color)
        sum_log_val = sum_logits.item()
        candidates.append((sum_log_val, action_tuple))
        created_children += 1

    if not candidates:
        return

    candidates.sort(key=lambda x: x[0], reverse=True)
    top_expansions = candidates[:expansions_per_node]

    top_logs = [x[0] for x in top_expansions]
    best_sum_log = top_logs[0]
    exps = [math.exp(val - best_sum_log) for val in top_logs]
    total_exp = sum(exps)

    for i, (sum_log_val, action_tuple) in enumerate(top_expansions):
        child_node = MCTSNode(node.state.clone())
        child_node.state.apply_action(action_tuple)

        local_prob = exps[i] / (total_exp + 1e-8)
        child_node.prior_prob = local_prob
        node.children[action_tuple] = child_node

def evaluate_and_expand_node(node, net, expansions_per_node, num_candidates):
    with torch.no_grad():
        out = net.forward(
            [
                {
                    "puzzle_id": node.state.puzzle_id,
                    "demo_input": node.state.demo_input,
                    "current_guess": node.state.current_guess,
                }
            ]
        )

    expand_node(
        node=node,
        row_logits=out["row_logits"],
        col_logits=out["col_logits"],
        width_logits=out["width_logits"],
        height_logits=out["height_logits"],
        color_logits=out["color_logits"],
        net_value_est=out["value"],
        net_done_logits=out["done_logits"],
        expansions_per_node=expansions_per_node,
        num_candidates=num_candidates,
    )
    return out["value"]

def run_mcts(
    root_state,
    net,
    n_sims,
    expansions_per_node,
    max_depth,
    num_candidates,
    c_puct,
    step_penalty,
):
    """
    root_state: a PuzzleState
    net: a model returning row_logits, col_logits, width_logits, height_logits, color_logits, and value
    n_sims: number of MCTS rollouts
    expansions_per_node: how many new actions to sample at each expansion
    max_depth: maximum depth of the MCTS tree
    num_candidates: how many candidates we consider sampling at each node
    c_puct: exploration constant
    step_penalty: penalty for each step taken
    """
    root_node = MCTSNode(root_state)
    if root_state.is_terminal():
        # No expansions to do
        root_node.visit_count = 1
        root_node.total_value = root_state.get_reward()
        pi_mcts = {}
        return pi_mcts, root_node

    # 1) Evaluate root node once
    evaluate_and_expand_node(
        root_node,
        net,
        expansions_per_node,
        num_candidates,
    )

    # 2) Run simulations
    for _ in range(n_sims):
        node = root_node
        search_path = [node]

        # (a) Select
        depth = 0
        while depth < max_depth:
            if node.state.is_terminal():
                break
            if len(node.children) == 0:  # unexpanded leaf
                break
            _, child = select_child(node, c_puct=c_puct)
            node = child
            search_path.append(node)
            depth += 1

        leaf_node = search_path[-1]
        leaf_state = leaf_node.state

        if leaf_state.is_terminal():
            leaf_node.debug_eventual_solution_found = True

        # (b) Expand leaf
        if not leaf_state.is_terminal():
            evaluate_and_expand_node(
                leaf_node,
                net,
                expansions_per_node,
                num_candidates,
                should_use_fake_actions=False,
            )

        leaf_node.visit_count += 1
        if leaf_node.state.is_terminal():
            leaf_node.total_value += leaf_state.get_reward()
        else:
            leaf_node.total_value += leaf_node.net_value_est

        # (c) Backup
        # The value is the sum of the current incremental reward and the estimate of future
        # incremental rewards.
        for j in reversed(range(len(search_path) - 1)):
            parent_node = search_path[j]
            child_node = search_path[j + 1]

            assert id(parent_node.state.current_guess) != id(
                child_node.state.current_guess
            )

            parent_reward = parent_node.state.get_reward()
            child_reward = child_node.state.get_reward()
            immediate_reward = (child_reward - parent_reward) - step_penalty

            if child_node.visit_count > 0:
                child_q = child_node.total_value / (child_node.visit_count + 1e-8)
            else:
                child_q = child_node.net_value_est
            backup_value = immediate_reward + child_q

            parent_node.visit_count += 1
            parent_node.total_value += backup_value

    # 3) Build pi_mcts from child visits at root
    pi_mcts = {}
    sum_visits = sum(child.visit_count for child in root_node.children.values())
    if sum_visits == 0:  # no expansions or visits
        for action_tuple in root_node.children.keys():
            pi_mcts[action_tuple] = 1.0 / len(root_node.children)
    else:
        for action_tuple, child_node in root_node.children.items():
            pi_mcts[action_tuple] = child_node.visit_count / sum_visits

    return pi_mcts, root_node
