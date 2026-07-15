import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def markov_denoise_pytorch(noisy_img, iterations=10, obs_weight=2.0, trans_weight=1.5, threshold=None):
    # Move tensor to GPU (using one of your 3090 Ti cards)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Format: [Batch, Channel, Height, Width]
    x = noisy_img.unsqueeze(0).unsqueeze(0).to(device).float()
    
    # Observation Factor: original noisy pixels as a 'bias'
    # o_i(x_i) = [x_i = observed value at i]
    observation = x.clone()
    
    # Transition Factor Kernel: 8-neighbors (1s on edges/corners, 0 in center)
    kernel = torch.ones((1, 1, 3, 3)).to(device)
    kernel[0, 0, 1, 1] = 0 
    
    # Adaptive threshold: if not provided, use (max possible neighbors / 2) + obs_weight/2
    if threshold is None:
        threshold = (8.0 / 2.0) + (obs_weight / 2.0)
    
    for _ in range(iterations):
        # Calculate neighbor agreement: sum of neighboring 1s
        neighbor_agreement = F.conv2d(x, kernel, padding=1)
        
        # Total Weight(x_i=1) = Observation_Weight + (Transition_Weight * Neighbor_Sum)
        # We compare this against a threshold (representing the weight of being 0)
        total_weight_1 = (obs_weight * observation) + (trans_weight * neighbor_agreement)
        
        # Threshold: if Weight for 1 > threshold, stay 1
        # This simulates the marginal probability logic from your Stanford notes
        x = (total_weight_1 > threshold).float()
        
    return x.squeeze().cpu()

def markov_denoise_grayscale(noisy_img, iterations=10, obs_weight=0.3, trans_weight=0.7):
    """Denoise grayscale image directly without binarization."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Normalize to [0, 1] range
    x = noisy_img.unsqueeze(0).unsqueeze(0).to(device).float()
    if x.max() > 1.0:
        x = x / 255.0
    
    observation = x.clone()
    
    # Gaussian-like smoothing kernel (weights neighbors by distance)
    kernel = torch.tensor([
        [0.0625, 0.125, 0.0625],
        [0.125,  0.0,   0.125],
        [0.0625, 0.125, 0.0625]
    ]).unsqueeze(0).unsqueeze(0).to(device)
    
    for _ in range(iterations):
        # Calculate weighted neighbor average
        neighbor_avg = F.conv2d(x, kernel, padding=1)
        
        # Blend observation with neighbor smoothing
        x = obs_weight * observation + trans_weight * neighbor_avg
        
        # Clamp to valid range
        x = torch.clamp(x, 0.0, 1.0)
        
    return x.squeeze().cpu()

def forward_backward_denoise(noisy_img, obs_weight=2.0, trans_weight=1.0):
    """
    Apply Forward-Backward algorithm for image denoising.
    Treats each row/column as a sequence and computes exact marginals.
    
    Args:
        noisy_img: Binary image tensor (H, W) with values 0 or 1
        obs_weight: Weight for observation factor
        trans_weight: Weight for transition factor (smoothness)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure binary input - use adaptive thresholding to preserve more detail
    x = noisy_img.to(device).float()
    if x.max() > 1.0:
        # Use mean-based thresholding (much faster than median for large images)
        # Mean is a good approximation and computes quickly on GPU
        print("Using adaptive thresholding (mean-based)...")
        threshold = x.mean().item()  # Mean is faster than median
        print(f"Threshold: {threshold:.1f}")
        x = (x >= threshold).float()
    
    H, W = x.shape
    print(f"Image shape: {H}x{W}")
    num_states = 2  # Binary: 0 or 1
    
    def forward_backward_1d(sequence, obs_weight, trans_weight):
        """Apply forward-backward to a 1D sequence"""
        T = sequence.shape[0]
        
        # Transition matrix: [prev_state, curr_state] -> weight
        # Prefer same state (smoothness)
        # trans_matrix[0][0] = 2.0 → Strong preference: 0 → 0 (staying black)
        # trans_matrix[0][1] = 1.0 → Weaker preference: 0 → 1 (black to white)
        # trans_matrix[1][0] = 1.0 → Weaker preference: 1 → 0 (white to black)
        # trans_matrix[1][1] = 2.0 → Strong preference: 1 → 1 (staying white)
        trans_matrix = torch.tensor([
            [trans_weight * 2.0, trans_weight * 1.0],  # from state 0
            [trans_weight * 1.0, trans_weight * 2.0]   # from state 1
        ], device=device)
        
        # Forward messages: F[i][s] = sum of weights of paths from start to position i with state s
        F = torch.zeros((T, num_states), device=device)
        
        # Initialize: observation factor at first position (vectorized)
        states = torch.arange(num_states, device=device, dtype=sequence.dtype)
        obs_mask = (sequence[0] == states).float()
        F[0] = obs_weight * obs_mask
        
        # Forward pass: vectorized for all states at once
        for i in range(1, T):
            # Compute forward sums for all states simultaneously
            forward_sums = (F[i-1].unsqueeze(1) * trans_matrix).sum(dim=0)  # Shape: (num_states,)
            
            # Observation factors for all states (vectorized)
            obs_mask = (sequence[i] == states).float()
            obs_factors = obs_weight * obs_mask
            
            F[i] = forward_sums * obs_factors
        
        # Backward messages: B[i][s] = sum of weights from position i to end given state s
        B = torch.ones((T, num_states), device=device)
        
        # Backward pass: vectorized
        for i in range(T-2, -1, -1):
            # Observation factors for next position (all states)
            obs_mask = (sequence[i+1] == states).float()
            obs_factors = obs_weight * obs_mask
            
            # Compute backward sums for all states simultaneously
            # B[i+1] * trans_matrix * obs_factors, then sum over next states
            backward_sums = (B[i+1].unsqueeze(0) * trans_matrix * obs_factors.unsqueeze(0)).sum(dim=1)
            B[i] = backward_sums
        
        # Combined: S[i][s] = F[i][s] * B[i][s] (unnormalized marginal)
        S = F * B
        
        # Normalize to get probabilities
        S_sum = S.sum(dim=1, keepdim=True)
        S_normalized = S / (S_sum + 1e-10)
        
        # Return most likely state (MAP estimate)
        return S_normalized.argmax(dim=1).float()
    
    # Apply forward-backward row by row
    print("Applying forward-backward row by row...")
    result_rows = torch.zeros_like(x)
    for row in range(H):
        if (row + 1) % max(1, H // 10) == 0 or row == 0:
            print(f"  Row progress: {row + 1}/{H} ({100*(row+1)//H}%)")
        result_rows[row] = forward_backward_1d(x[row], obs_weight, trans_weight)
    
    # Apply forward-backward column by column
    print("Applying forward-backward column by column...")
    result_cols = torch.zeros_like(x)
    for col in range(W):
        if (col + 1) % max(1, W // 10) == 0 or col == 0:
            print(f"  Column progress: {col + 1}/{W} ({100*(col+1)//W}%)")
        result_cols[:, col] = forward_backward_1d(x[:, col], obs_weight, trans_weight)
    
    # Combine row and column results (average)
    result = (result_rows + result_cols) / 2.0
    result = (result > 0.5).float()  # Threshold to binary
    
    return result.cpu()

def forward_backward_denoise_grayscale(noisy_img, obs_weight=0.3, trans_weight=0.7, num_levels=8):
    """
    Apply Forward-Backward algorithm for grayscale image denoising.
    Uses quantized intensity levels to preserve more detail than binary version.
    
    Args:
        noisy_img: Grayscale image tensor (H, W) with values 0-255
        obs_weight: Weight for observation factor
        trans_weight: Weight for transition factor (smoothness)
        num_levels: Number of intensity levels to use (8, 16, or 32 for balance)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Normalize and quantize input
    x = noisy_img.to(device).float()
    if x.max() > 1.0:
        x = x / 255.0  # Normalize to [0, 1]
    
    # Quantize to num_levels discrete levels
    x_quantized = (x * (num_levels - 1)).round().long()
    H, W = x_quantized.shape
    
    def forward_backward_1d_grayscale(sequence, obs_weight, trans_weight, num_levels):
        """Apply forward-backward to a 1D sequence with multiple intensity levels"""
        T = sequence.shape[0]
        sequence = sequence.long()  # Ensure integer type
        
        # Transition matrix: prefer similar intensity levels (smoothness)
        # Vectorized computation - much faster
        states = torch.arange(num_levels, device=device).float()
        i_grid, j_grid = torch.meshgrid(states, states, indexing='ij')
        dist = torch.abs(i_grid - j_grid)
        trans_matrix = torch.where(dist == 0, trans_weight * 2.0,
                          torch.where(dist == 1, trans_weight * 1.0,
                          trans_weight / (dist + 1)))
        
        # Forward messages
        F = torch.zeros((T, num_levels), device=device)
        
        # Initialize: vectorized observation factor
        obs_value = sequence[0].float()
        dist_to_obs = torch.abs(states - obs_value)
        obs_factors = obs_weight / (dist_to_obs + 1)
        F[0] = obs_factors
        
        # Forward pass - vectorized
        for i in range(1, T):
            # Compute forward sum for all states at once
            forward_sums = (F[i-1].unsqueeze(1) * trans_matrix).sum(dim=0)  # Shape: (num_levels,)
            
            # Observation factors for all states
            obs_value = sequence[i].float()
            dist_to_obs = torch.abs(states - obs_value)
            obs_factors = obs_weight / (dist_to_obs + 1)
            
            F[i] = forward_sums * obs_factors
        
        # Backward messages
        B = torch.ones((T, num_levels), device=device)
        
        # Backward pass - partially vectorized
        for i in range(T-2, -1, -1):
            obs_value = sequence[i+1].float()
            dist_to_obs = torch.abs(states - obs_value)
            obs_factors = obs_weight / (dist_to_obs + 1)  # Shape: (num_levels,)
            
            # Vectorized: compute backward sum for all states
            # B[i+1] * trans_matrix[s, :] * obs_factors for each s
            backward_sums = (B[i+1].unsqueeze(0) * trans_matrix * obs_factors.unsqueeze(0)).sum(dim=1)
            B[i] = backward_sums
        
        # Combined: get most likely intensity level
        S = F * B
        S_sum = S.sum(dim=1, keepdim=True)
        S_normalized = S / (S_sum + 1e-10)
        
        # Return most likely level
        return S_normalized.argmax(dim=1).float()
    
    # Apply forward-backward row by row
    print(f"Applying grayscale forward-backward ({num_levels} levels) row by row...")
    result_rows = torch.zeros_like(x_quantized, dtype=torch.float32)
    for row in range(H):
        if (row + 1) % max(1, H // 10) == 0:
            print(f"  Row progress: {row + 1}/{H}")
        result_rows[row] = forward_backward_1d_grayscale(x_quantized[row], obs_weight, trans_weight, num_levels)
    
    # Apply forward-backward column by column
    print("Applying grayscale forward-backward column by column...")
    result_cols = torch.zeros_like(x_quantized, dtype=torch.float32)
    for col in range(W):
        if (col + 1) % max(1, W // 10) == 0:
            print(f"  Column progress: {col + 1}/{W}")
        result_cols[:, col] = forward_backward_1d_grayscale(x_quantized[:, col], obs_weight, trans_weight, num_levels)
    
    # Combine and convert back to [0, 1] range
    result = (result_rows + result_cols) / 2.0
    result = result / (num_levels - 1)  # Normalize back to [0, 1]
    
    return result.cpu()

def load_and_preprocess_image(image_path, threshold=128):
    """Load an image, convert to grayscale, and binarize it."""
    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)
    
    # Binarize: values >= threshold become 1, else 0
    binary_img = (img_array >= threshold).astype(np.float32)
    
    # Convert to torch tensor
    return torch.from_numpy(binary_img), img_array

# Load image (update path to your noisy image)
image_path = '../data/noisy_image.png'  # Change this to your image path

# Load and preprocess the image
noisy_img, original_gray = load_and_preprocess_image(image_path)

# Try different denoising approaches
print("Denoising with improved binary approach...")
denoised_binary = markov_denoise_pytorch(noisy_img, iterations=15, obs_weight=2.5, trans_weight=2.0)

print("Denoising with grayscale approach...")
original_tensor = torch.from_numpy(original_gray)
denoised_grayscale = markov_denoise_grayscale(original_tensor, iterations=20, obs_weight=0.2, trans_weight=0.8)

print("Applying median filter (good for salt-and-pepper noise)...")
denoised_median = ndimage.median_filter(original_gray, size=3)

print("Applying Forward-Backward algorithm (binary)...")
denoised_fb = forward_backward_denoise(noisy_img, obs_weight=2.0, trans_weight=1.5)

print("Applying Forward-Backward algorithm (grayscale, preserves detail)...")
original_tensor = torch.from_numpy(original_gray)
# Use 8 levels for faster computation (can increase to 16 for more detail)
denoised_fb_grayscale = forward_backward_denoise_grayscale(original_tensor, obs_weight=0.3, trans_weight=0.7, num_levels=8)

# Convert to numpy for display
denoised_binary_np = denoised_binary.numpy()
denoised_grayscale_np = denoised_grayscale.numpy() * 255.0  # Convert back to 0-255 range
denoised_fb_np = denoised_fb.numpy()
denoised_fb_grayscale_np = denoised_fb_grayscale.numpy() * 255.0  # Convert back to 0-255 range

# Display results - show 7 panels in 2 rows (4 in first row, 3 in second)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].imshow(original_gray, cmap='gray')
axes[0, 0].set_title('Original Grayscale Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(noisy_img.numpy(), cmap='gray')
axes[0, 1].set_title('Binarized (Noisy) Input')
axes[0, 1].axis('off')

axes[0, 2].imshow(denoised_median, cmap='gray')
axes[0, 2].set_title('Median Filter (3x3)')
axes[0, 2].axis('off')

axes[0, 3].imshow(denoised_fb_np, cmap='gray')
axes[0, 3].set_title('Forward-Backward (Binary)')
axes[0, 3].axis('off')

axes[1, 0].imshow(denoised_fb_grayscale_np, cmap='gray')
axes[1, 0].set_title('Forward-Backward (Grayscale, 16 levels)')
axes[1, 0].axis('off')

axes[1, 1].imshow(denoised_binary_np, cmap='gray')
axes[1, 1].set_title('Denoised (Binary, Improved)')
axes[1, 1].axis('off')

axes[1, 2].imshow(denoised_grayscale_np, cmap='gray')
axes[1, 2].set_title('Denoised (Grayscale Approach)')
axes[1, 2].axis('off')

# Combine: apply median filter then Markov denoising
print("Combining median filter + Markov denoising...")
median_binary = (denoised_median >= 128).astype(np.float32)
median_tensor = torch.from_numpy(median_binary)
denoised_combined = markov_denoise_pytorch(median_tensor, iterations=10, obs_weight=2.0, trans_weight=1.5)
axes[1, 3].imshow(denoised_combined.numpy(), cmap='gray')
axes[1, 3].set_title('Median + Markov Denoising')
axes[1, 3].axis('off')

plt.tight_layout()
plt.show()