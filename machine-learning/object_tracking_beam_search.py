import torch

def beam_search_hmm(K=3):
    # Observation weights (oi) from the Stanford tracking example
    # Steps: X1, X2, X3 | Positions: 0, 1, 2
    obs_weights = [
        {0: 2, 1: 1, 2: 0},  # o1
        {0: 0, 1: 1, 2: 2},  # o2
        {0: 0, 1: 1, 2: 2}   # o3
    ]

    # Transition weights (ti): distance 0 -> 2, dist 1 -> 1, dist 2 -> 0
    def get_trans_weight(prev_v, curr_v):
        dist = abs(prev_v - curr_v)
        if dist == 0: return 2
        if dist == 1: return 1
        return 0

    # Initialize candidate list C with an empty assignment and weight 1.0
    candidates = [([], 1.0)]

    for i in range(len(obs_weights)):
        extensions = []
        
        # EXTEND phase
        for path, path_weight in candidates:
            for v in [0, 1, 2]:  # Domain for Hi
                # Calculate new weight: Previous weight * Transition * Observation
                tw = get_trans_weight(path[-1], v) if path else 1
                ow = obs_weights[i][v]
                
                new_weight = path_weight * tw * ow
                if new_weight > 0:
                    extensions.append((path + [v], new_weight))
        
        # PRUNE phase: Keep only K particles with highest weights
        extensions.sort(key=lambda x: x[1], reverse=True)
        candidates = extensions[:K]

    return candidates

# Execute on your AI server environment
print("Running beam search on CPU...")
results = beam_search_hmm(K=3)
for path, weight in results:
    print(f"Path: {path} | Total Weight: {weight}")