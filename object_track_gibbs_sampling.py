import random
import torch
from collections import Counter

# Factors based on the Stanford slides
obs_factors = {
    'X1': {0: 2, 1: 1, 2: 0},
    'X2': {0: 0, 1: 1, 2: 2},
    'X3': {0: 0, 1: 1, 2: 2}
}

def transition_weight(v1, v2):
    diff = abs(v1 - v2)
    if diff == 0: return 2
    if diff == 1: return 1
    return 0

def run_gibbs_tracking(iterations=10000):
    # 1. Initialize variables randomly
    state = {'X1': 0, 'X2': 1, 'X3': 2}
    history = {'X1': [], 'X2': [], 'X3': []}
    
    variables = ['X1', 'X2', 'X3']
    
    for _ in range(iterations):
        for i, var in enumerate(variables):
            weights = []
            for val in [0, 1, 2]:
                # Calculate local weight for this variable value
                w = obs_factors[var][val]
                
                # Multiply by transition to PREVIOUS variable
                if i > 0:
                    prev_var = variables[i-1]
                    w *= transition_weight(state[prev_var], val)
                
                # Multiply by transition to NEXT variable
                if i < len(variables) - 1:
                    next_var = variables[i+1]
                    w *= transition_weight(val, state[next_var])
                
                weights.append(w)
            
            # 2. The "Dice Roll" (weighted sampling)
            if sum(weights) > 0:
                state[var] = random.choices([0, 1, 2], weights=weights)[0]
            
            # 3. Save the state for estimating marginals
            history[var].append(state[var])
            
    return history


def gibbs_tracking_pytorch(steps=10, iterations=10000):
    # Use one of your 3090 Ti cards
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Factors (using the logic from your Stanford slides)
    # obs_map: weight for [pos 0, pos 1, pos 2] at each time step
    # Example: X1 favors 0, X2 favors 2, X3 favors 2...
    obs_weights = torch.tensor([
        [2, 1, 0], [0, 1, 2], [0, 1, 2], [0, 1, 1], [1, 2, 1],
        [2, 1, 0], [1, 2, 1], [0, 1, 2], [0, 1, 2], [1, 1, 1]
    ], dtype=torch.float32, device=device)

    # Initialize positions randomly: Shape (steps,)
    state = torch.randint(0, 3, (steps,), device=device)
    
    # Store counts for marginal probabilities: Shape (steps, 3 positions)
    counts = torch.zeros((steps, 3), device=device)
    
    # Pre-create v_options tensor (reused in loop)
    v_options = torch.tensor([0, 1, 2], device=device, dtype=torch.long)

    for i in range(iterations):
        for t in range(steps):
            # 1. Observation Weight
            weights = obs_weights[t].clone()
            
            # 2. Transition Weight from PREVIOUS (if exists)
            if t > 0:
                # |v - state[t-1]|: dist 0 -> weight 2, dist 1 -> weight 1, else 0
                dist_prev = torch.abs(v_options - state[t-1])
                t_prev = torch.where(dist_prev == 0, 2.0, torch.where(dist_prev == 1, 1.0, 0.0))
                weights *= t_prev
                
            # 3. Transition Weight to NEXT (if exists)
            if t < steps - 1:
                dist_next = torch.abs(v_options - state[t+1])
                t_next = torch.where(dist_next == 0, 2.0, torch.where(dist_next == 1, 1.0, 0.0))
                weights *= t_next
            
            # The "Dice Roll": Sample from unnormalized weights
            if weights.sum() > 0:
                # torch.multinomial returns a tensor of shape [1], extract the value
                state[t] = torch.multinomial(weights, 1)[0]
            
            # Accumulate counts for marginals (skip 'burn-in' first 1000)
            if i > 1000:
                counts[t, state[t]] += 1
        
        # Progress output every 1000 iterations
        if (i + 1) % 1000 == 0:
            print(f"Progress: {i + 1}/{iterations} iterations completed")
                
    return counts / (iterations - 1000)

# Run the simulation
print("Running Gibbs sampling on CPU...")
results = run_gibbs_tracking()
print("Results:")
for var in ['X1', 'X2', 'X3']:
    counts = Counter(results[var])
    print(f"Marginal P({var}):", {k: v/10000 for k, v in counts.items()})

# Run and print X2 marginals
# print("Running Gibbs sampling on GPU...")
# print("Note: This may take a minute with 10,000 iterations...")
# marginals = gibbs_tracking_pytorch(iterations=10000)  # Can reduce to 1000 for faster testing
# print("Results:")
# print(f"X1 Marginals (Pos 0, 1, 2): {marginals[0].cpu().numpy()}")
# print(f"X2 Marginals (Pos 0, 1, 2): {marginals[1].cpu().numpy()}")
# print(f"X3 Marginals (Pos 0, 1, 2): {marginals[2].cpu().numpy()}")  