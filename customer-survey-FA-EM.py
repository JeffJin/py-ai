import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# ==========================================
# STEP 1: GENERATE SYNTHETIC DATA (GROUND TRUTH)
# ==========================================
n_samples = 500
d_features = 6  # 6 survey questions
k_latents = 2  # 2 underlying mental states

# True factor loadings matrix (6 features x 2 factors)
# Q0-Q2 load heavily on Factor 1; Q3-Q5 load heavily on Factor 2
true_Lambda = np.array([
  [0.9, 0.1],  # Q0
  [0.8, 0.0],  # Q1
  [0.85, 0.15],  # Q2
  [0.1, 0.75],  # Q3
  [0.0, 0.9],  # Q4
  [0.2, 0.8]  # Q5
])

# True diagonal unique variance (noise)
true_Psi_diag = np.array([0.19, 0.36, 0.25, 0.43, 0.19, 0.32])  # 1 - variance of rows
true_Psi = np.diag(true_Psi_diag)

# Generate data following: x = Lambda * z + epsilon
z_true = np.random.multivariate_normal(np.zeros(k_latents), np.eye(k_latents), size=n_samples)
epsilon_true = np.random.multivariate_normal(np.zeros(d_features), true_Psi, size=n_samples)
X = z_true @ true_Lambda.T + epsilon_true

print(f"Generated data matrix X with shape: {X.shape}")

# ==========================================
# STEP 2: INITIALIZE EM PARAMETERS
# ==========================================
# Step 2a: Empirical mean calculation and data centering
mu_hat = np.mean(X, axis=0)
X_centered = X - mu_hat

# Step 2b: Randomly initialize model targets
Lambda_hat = np.random.randn(d_features, k_latents) * 0.1
Psi_diag_hat = np.var(X_centered, axis=0)  # start with total variance as noise guess

# ==========================================
# STEP 3: THE EM OPTIMIZATION LOOP
# ==========================================
max_iters = 100
tol = 1e-5
prev_ll = -np.inf

print("\nStarting EM Training Loop...")
for iteration in range(max_iters):

  # ------------------
  # E-STEP
  # ------------------
  # Compute shared matrix inversion component common to all steps
  # Sigma_xx = Lambda * Lambda^T + Psi
  Sigma_xx = Lambda_hat @ Lambda_hat.T + np.diag(Psi_diag_hat)
  Sigma_xx_inv = np.linalg.inv(Sigma_xx)

  # Pre-allocate arrays to hold expected statistics for the M-step
  E_z = np.zeros((n_samples, k_latents))
  sum_E_zz = np.zeros((k_latents, k_latents))

  # Common conditional covariance across ALL points (Equation 2 clone)
  # Sigma_z|x = I - Lambda^T * Sigma_xx_inv * Lambda
  Sigma_z_given_x = np.eye(k_latents) - Lambda_hat.T @ Sigma_xx_inv @ Lambda_hat

  for i in range(n_samples):
    xi = X_centered[i]

    # Calculate conditional mean for sample i (Equation 1 clone)
    # mu_z|x = Lambda^T * Sigma_xx_inv * (x_i)
    mu_zi = Lambda_hat.T @ Sigma_xx_inv @ xi
    E_z[i] = mu_zi

    # Calculate second moment for sample i (Identity substitution)
    # E[z z^T] = mu * mu^T + Sigma
    sum_E_zz += np.outer(mu_zi, mu_zi) + Sigma_z_given_x

  # ------------------
  # M-STEP
  # ------------------
  # Update Lambda Matrix using the analytical regression framework (Equation 8)
  # Lambda_new = (Sum_i x_i * E[z_i]^T) * (Sum_i E[z_i z_i^T])^-1
  matrix_A = X_centered.T @ E_z
  Lambda_hat = matrix_A @ np.linalg.inv(sum_E_zz)

  # Update unique variance diagonal (Phi / Psi tracking)
  # We apply the full expectation formula variance profile expansion point-by-point
  # and extract only the diagonal entries.
  Lambda_sum_E_zz_LambdaT = Lambda_hat @ sum_E_zz @ Lambda_hat.T
  X_cross_Ez_LambdaT = X_centered.T @ E_z @ Lambda_hat.T

  # Total accumulated variance residual matrix
  Psi_full = (
                   X_centered.T @ X_centered - X_cross_Ez_LambdaT - X_cross_Ez_LambdaT.T + Lambda_sum_E_zz_LambdaT) / n_samples
  Psi_diag_hat = np.diag(Psi_full)

  # ------------------
  # LOG-LIKELIHOOD EVALUATION
  # ------------------
  # Monitor convergence using the exact marginalized score formula we mapped
  sign, logdet = np.linalg.slogdet(Sigma_xx)
  ll = 0
  for i in range(n_samples):
    dist = X_centered[i] @ Sigma_xx_inv @ X_centered[i]
    ll += -0.5 * (d_features * np.log(2 * np.pi) + logdet + dist)

  if np.abs(ll - prev_ll) < tol:
    print(f"Converged early at iteration {iteration}. Log-Likelihood: {ll:.4f}")
    break

  prev_ll = ll
  if iteration % 20 == 0:
    print(f"Iteration {iteration:02d} | Log-Likelihood: {ll:.4f}")

# ==========================================
# STEP 4: VERIFY RESULTS
# ==========================================
print("\n=== ESTIMATION VERIFICATION ===")
print("\nTrue Lambda Loading Structure:")
print(np.round(true_Lambda, 3))

print("\nEstimated Lambda Loading Structure (Discovered by EM):")
print(np.round(np.abs(Lambda_hat), 3))
# Note: np.abs accounts for arbitrary global factor sign flips