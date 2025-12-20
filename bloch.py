import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib as mpl
import time

def profile_training_speed():
    """Run this ONCE before training to see bottlenecks"""
    print("\n" + "="*70)
    print("PROFILING...")
    print("="*70)
    
    model_test = BlochPINN(B=1).to(device)
    batch_size = 2500
    
    times = {}
    
    # Time each component
    t = time.time()
    x, y = sample_cell(batch_size, a1, a2)
    kx, ky = sample_IBZ(batch_size)
    uR, uI, E = model_test(x, y, kx, ky)
    loss_pde = bloch_pde_loss(uR, uI, E, x, y, kx, ky, honeycomb_potential)
    times['PDE'] = time.time() - t
    
    t = time.time()
    kx_norm, ky_norm = sample_IBZ(100)
    loss_norm = norm_loss(model_test, kx_norm, ky_norm, a1, a2, grid_n=50)
    times['Norm'] = time.time() - t
    
    t = time.time()
    kx_bc, ky_bc = sample_IBZ(64)
    loss_bc = lattice_bc_loss(model_test, kx_bc, ky_bc, a1, a2, n_pts=500)
    times['BC'] = time.time() - t
    
    total_time = sum(times.values())
    
    print("\nTime breakdown:")
    for name, tm in sorted(times.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * tm / total_time
        print(f"  {name:10s}: {tm*1000:6.1f} ms ({pct:4.1f}%)")
    
    print(f"\n  Est. 10k epochs: {total_time*10000/60:.1f} minutes")
    print("="*70 + "\n")

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 14,
    "font.size": 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "text.latex.preamble": r"\usepackage{amsmath,amssymb}"
})

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
x_test = torch.tensor([0.5], device=device)
print(f"Device test OK: {x_test.device}")

a = 1.0
q = torch.tensor(4*np.pi / (np.sqrt(3)*a), dtype=torch.float32, device=device)

a1 = torch.tensor([a*np.sqrt(3)/2,  a*1/2], dtype=torch.float32, device=device)
a2 = torch.tensor([a*np.sqrt(3)/2, -a*1/2], dtype=torch.float32, device=device)

# Reciprocal vectors
k1 = torch.tensor([0.5*q,  np.sqrt(3)/2*q], dtype=torch.float32, device=device)
k2 = torch.tensor([0.5*q, -np.sqrt(3)/2*q], dtype=torch.float32, device=device)

# High symmetry points
Gamma = torch.zeros(1, 2, dtype=torch.float32, device=device)   # (1,2)
K     = ((2*k1 + k2) / 3.0).reshape(1,2)                        # (1,2)
M     = ((k1 + k2) / 2.0).reshape(1,2)                           # (1,2)

kx_G = Gamma[:,0:1]
ky_G = Gamma[:,1:2]

kx_K = K[:,0:1]
ky_K = K[:,1:2]

kx_M = M[:,0:1]
ky_M = M[:,1:2]

# Reciprocal magnitude
G = torch.tensor(4*np.pi / (np.sqrt(3)*a), dtype=torch.float32, device=device)

# Precomputed reciprocal lattice vectors
K1 = torch.tensor([0.5*G,  np.sqrt(3)/2*G], dtype=torch.float32, device=device)
K2 = torch.tensor([0.5*G, -np.sqrt(3)/2*G], dtype=torch.float32, device=device)
K3 = torch.tensor([G,      0.0          ], dtype=torch.float32, device=device)

def honeycomb_potential(x, y, V0=1.0):
    """
    Smooth C6-symmetric honeycomb potential via 3 plane waves.
    Works for x,y of shape (N,1) or (n,n).
    """
    r = torch.stack((x, y), dim=-1).to(torch.float32)

    return V0 * (
        torch.cos(r @ K1) +
        torch.cos(r @ K2) +
        torch.cos(r @ K3)
    )
    
# New potential with V0=10
def honeycomb_potential_v10(x, y, V0=10.0):
    r = torch.stack((x, y), dim=-1).to(torch.float32)
    return V0 * (
        torch.cos(r @ K1) +
        torch.cos(r @ K2) +
        torch.cos(r @ K3)
    )

def D(var, f):
    """First derivative df/dvar using autograd."""
    return torch.autograd.grad(
        f, var, 
        grad_outputs=torch.ones_like(f),
        create_graph=True
    )[0]

def D2(var, f):
    """Second derivative ddf/dvardvar."""
    df = D(var, f)
    return torch.autograd.grad(
        df, var, 
        grad_outputs=torch.ones_like(df),
        create_graph=True
    )[0]
    

def laplacian(psi, x, y):
    return D2(x, psi) + D2(y, psi)

class BlochPINN(nn.Module):
    """
    Multi-band Bloch PINN:
    - bloch net learns periodic Bloch amplitude u_b(x,y;k)
    - energy net learns band energies E_b(k)
    """
    def __init__(self, B=1, f_width=200, f_depth=5,
                 e_width=64, e_depth=3):
        super().__init__()
        self.B = B
        
        #energy net
        self.e_in = nn.Linear(2, e_width)  # (kx,ky)
        self.e_hidden = nn.ModuleList(
            [nn.Linear(e_width, e_width) for _ in range(e_depth)]
        )
        self.e_out = nn.Linear(e_width, B)  # predicts B energies
        
        self.e_act = nn.SiLU()
        
        #bloch net
        self.f_in = nn.Linear(4, f_width)  # (x,y,kx,ky)
        self.f_hidden = nn.ModuleList(
            [nn.Linear(f_width, f_width) for _ in range(f_depth)]
        )
        self.f_out = nn.Linear(f_width, 2*B)  # (uR,uI) for each band
        
        self.f_act = nn.SiLU()

    def forward(self, x, y, kx, ky):
        """
        Returns:
            uR, uI: (N,B) periodic Bloch amplitudes
            E:     (N,B) energy values at k
        """
        # pass energynet
        k = torch.cat([kx, ky], dim=1)   # (N,2)
        e = self.e_act(self.e_in(k))
        for layer in self.e_hidden:
            e = self.e_act(layer(e))
        E = self.e_out(e)                # (N,B)
        
        # pass blochnet
        inp = torch.cat([x, y, kx, ky], dim=1)  # (N,4)
        f = self.f_act(self.f_in(inp))
        for layer in self.f_hidden:
            f = self.f_act(layer(f))
        
        out = self.f_out(f)              # (N,2B)
        out = out.view(-1, self.B, 2)    # (N,B,2)
        
        uR = out[:,:,0]                  # (N,B)
        uI = out[:,:,1]                  # (N,B)
        
        return uR, uI, E
    
def bloch_pde_loss(uR, uI, E, x, y, kx, ky, V_fn):
    """
    PDE residual without complex autograd:
    compute psi_real, psi_imag separately and take laplacian of each (real tensor).
    """
    Vxy = V_fn(x, y)                      # (N,1) or (N,1,?) but should broadcast to (N,1)

    k_dot_r = kx * x + ky * y             # (N,1)
    coskr = torch.cos(k_dot_r)
    sinkr = torch.sin(k_dot_r)

    # psi = (uR + i uI) e^{i k·r}
    psiR = uR * coskr - uI * sinkr        # (N,B)
    psiI = uR * sinkr + uI * coskr        # (N,B)

    B = uR.shape[1]
    loss = 0.0

    for b in range(B):
        psiRb = psiR[:, b:b+1]            # (N,1)
        psiIb = psiI[:, b:b+1]            # (N,1)
        Eb    = E[:,  b:b+1]              # (N,1)

        lapR = laplacian(psiRb, x, y)     # real laplacian
        lapI = laplacian(psiIb, x, y)

        rR = Eb * psiRb + 0.5 * lapR - Vxy * psiRb
        rI = Eb * psiIb + 0.5 * lapI - Vxy * psiIb

        loss += (rR.pow(2) + rI.pow(2)).mean()

    return loss / B

def norm_loss(model, kx, ky, a1, a2, grid_n=100):
    """
    - build a single big batch of (k × grid)
    - forward once
    - reshape into (n_k, grid_n, grid_n, B)
    - apply torch.trapezoid twice
    - enforce ∫cell |u|^2 = 1
    """

    # Number of k-points
    N_k = kx.shape[0]

    # ----------- Build spatial grid on [0,1]^2 -----------
    s = torch.linspace(0, 1, grid_n, device=device)
    s1, s2 = torch.meshgrid(s, s, indexing='ij')  # (grid_n, grid_n)

    # Map to physical unit cell: r = s1*a1 + s2*a2
    xg = s1 * a1[0] + s2 * a2[0]      # (grid_n, grid_n)
    yg = s1 * a1[1] + s2 * a2[1]      # (grid_n, grid_n)

    # Flatten spatial grid
    x_flat = xg.reshape(-1, 1)        # (G,1)
    y_flat = yg.reshape(-1, 1)
    G = grid_n * grid_n               # number of spatial points

    # Cell area
    A = torch.abs(a1[0]*a2[1] - a1[1]*a2[0])

    # ----------- Tile the grid for all k’s -----------
    # Expand kx, ky into (N_k * G, 1)
    kx_big = kx.repeat_interleave(G, dim=0)
    ky_big = ky.repeat_interleave(G, dim=0)

    # Expand x,y to match k count
    x_big = x_flat.repeat(N_k, 1)
    y_big = y_flat.repeat(N_k, 1)

    # ----------- SINGLE FORWARD PASS -----------
    uR, uI, _ = model(x_big, y_big, kx_big, ky_big)   # (N_k*G, B)
    B = uR.shape[1]

    # Compute |u|^2
    u2 = (uR**2 + uI**2).reshape(N_k, grid_n, grid_n, B)  # (N_k, gx, gy, B)

    # ----------- Apply 2D trapezoid-----------
    # First integrate along axis 2 (y)
    Iy = torch.trapezoid(u2, dx=1/grid_n, dim=2)    # (N_k, grid_n, B)

    # Then integrate along axis 1 (x)
    Ix = torch.trapezoid(Iy, dx=1/grid_n, dim=1)    # (N_k, B)

    # Convert ∫_{[0,1]^2} → ∫cell: multiply by cell area A
    integral = A * Ix                                # (N_k, B)

    loss = (integral - 1).pow(2).mean()

    return loss

def sample_cell(n_samples, a1, a2):
    """
    Sample points uniformly inside the parallelogram spanned by (a1,a2).
    """
    s1 = torch.rand(n_samples, 1, device=device)
    s2 = torch.rand(n_samples, 1, device=device)

    x = s1 * a1[0] + s2 * a2[0]
    y = s1 * a1[1] + s2 * a2[1]

    x.requires_grad_(True)
    y.requires_grad_(True)
    return x, y

def lattice_bc_loss(model, kx, ky, a1, a2, n_pts=2000):
    """
    Vectorized: enforce periodicity for all k-points at once
    """
    N_k = kx.shape[0]
    
    # ============ A1 PERIODICITY ============
    # Sample boundary points once
    t = torch.rand(n_pts//2, 1, device=device)
    
    x0 = t * a2[0]
    y0 = t * a2[1]
    x1 = x0 + a1[0]
    y1 = y0 + a1[1]
    
    # Tile for all k-points: (N_k * n_pts//2) points
    x0_all = x0.repeat(N_k, 1)
    y0_all = y0.repeat(N_k, 1)
    x1_all = x1.repeat(N_k, 1)
    y1_all = y1.repeat(N_k, 1)
    
    # Repeat k-values for each boundary point
    kx_bc = kx.repeat_interleave(n_pts//2, dim=0)
    ky_bc = ky.repeat_interleave(n_pts//2, dim=0)
    
    # Single forward pass for both sides
    x_a1 = torch.cat([x0_all, x1_all], dim=0)
    y_a1 = torch.cat([y0_all, y1_all], dim=0)
    kx_a1 = torch.cat([kx_bc, kx_bc], dim=0)
    ky_a1 = torch.cat([ky_bc, ky_bc], dim=0)
    
    uR_a1, uI_a1, _ = model(x_a1, y_a1, kx_a1, ky_a1)
    
    # Split results
    half = N_k * n_pts // 2
    uR0_a1, uR1_a1 = uR_a1[:half], uR_a1[half:]
    uI0_a1, uI1_a1 = uI_a1[:half], uI_a1[half:]
    
    L_a1 = ((uR1_a1 - uR0_a1)**2 + (uI1_a1 - uI0_a1)**2).mean()
    
    # ============ A2 PERIODICITY ============
    t = torch.rand(n_pts//2, 1, device=device)
    
    x0 = t * a1[0]
    y0 = t * a1[1]
    x1 = x0 + a2[0]
    y1 = y0 + a2[1]
    
    x0_all = x0.repeat(N_k, 1)
    y0_all = y0.repeat(N_k, 1)
    x1_all = x1.repeat(N_k, 1)
    y1_all = y1.repeat(N_k, 1)
    
    kx_bc = kx.repeat_interleave(n_pts//2, dim=0)
    ky_bc = ky.repeat_interleave(n_pts//2, dim=0)
    
    x_a2 = torch.cat([x0_all, x1_all], dim=0)
    y_a2 = torch.cat([y0_all, y1_all], dim=0)
    kx_a2 = torch.cat([kx_bc, kx_bc], dim=0)
    ky_a2 = torch.cat([ky_bc, ky_bc], dim=0)
    
    uR_a2, uI_a2, _ = model(x_a2, y_a2, kx_a2, ky_a2)
    
    uR0_a2, uR1_a2 = uR_a2[:half], uR_a2[half:]
    uI0_a2, uI1_a2 = uI_a2[:half], uI_a2[half:]
    
    L_a2 = ((uR1_a2 - uR0_a2)**2 + (uI1_a2 - uI0_a2)**2).mean()
    
    return L_a1 + L_a2

def sample_IBZ(n_k, radius=1):
    # barycentric sampling inside triangle
    s1 = torch.rand(n_k, 1, device=device)
    s2 = torch.rand(n_k, 1, device=device)

    mask = (s1 + s2 > 1)
    s1[mask], s2[mask] = 1 - s1[mask], 1 - s2[mask]

    k = Gamma + s1*(K-Gamma) + s2*(M-Gamma)

    # random reciprocal shifts
    nx = torch.randint(-radius, radius+1, (n_k, 1), device=device)
    ny = torch.randint(-radius, radius+1, (n_k, 1), device=device)

    k = k + nx*K1 + ny*K2
    return k[:,0:1], k[:,1:2]

def compute_grad_norm(loss, model):
    """Compute gradient norm of a loss w.r.t. model parameters"""
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, allow_unused=True)
    total_norm = 0.0
    for g in grads:
        if g is not None:
            total_norm += g.norm().item() ** 2
    return total_norm ** 0.5

def train_bloch_pinn(
    model,
    V_fn,
    a1, a2,
    n_epochs,
    batch_size=3000,
    lr=5e-4,
    alpha_bc=10,
    alpha_norm=100,
    grid_n=100,
    n_r_bc=300,
    rebalance=True,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=n_epochs, eta_min=1e-6
    )

    history = {"total": [], "pde": [], "norm": [], "bc": []}

    for epoch in range(n_epochs):

        # ------------------------------------------
        # 1) PDE loss (full 5000 batch)
        # ------------------------------------------
        x, y = sample_cell(batch_size, a1, a2)
        kx, ky = sample_IBZ(batch_size)

        uR, uI, E_batch = model(x, y, kx, ky)
        loss_pde = bloch_pde_loss(uR, uI, E_batch, x, y, kx, ky, V_fn)
        # ------------------------------------------
        # 2) Auxiliary losses use k-batches
        # ------------------------------------------
        kx_norm,  ky_norm  = sample_IBZ(100)
        kx_bc,    ky_bc    = sample_IBZ(50)

        loss_norm  = norm_loss(model,  kx_norm,  ky_norm,  a1, a2, grid_n=grid_n)
        loss_bc    = lattice_bc_loss(model, kx_bc, ky_bc, a1, a2, n_pts=n_r_bc)
        
        if epoch % 500 == 0 and epoch > 0 and rebalance:
            grad_pde = compute_grad_norm(loss_pde, model)
            grad_norm = compute_grad_norm(loss_norm, model)
            grad_bc = compute_grad_norm(loss_bc, model)
            
            # Target: make all gradient contributions roughly equal to PDE
            if grad_norm > 1e-8:
                alpha_norm = grad_pde / grad_norm
            if grad_bc > 1e-8:
                alpha_bc = grad_pde / grad_bc
            
            # Clamp to reasonable range
            alpha_norm = max(1.0, min(alpha_norm, 1000.0))
            alpha_bc = max(1.0, min(alpha_bc, 100.0))
            
            print(f"  [Rebalance] alpha_norm={alpha_norm:.1f}, alpha_bc={alpha_bc:.1f}")
        # ------------------------------------------
        # TOTAL LOSS
        # ------------------------------------------
        total = (
            loss_pde
            + alpha_norm  * loss_norm
            + alpha_bc    * loss_bc
        )

        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        scheduler.step()

        # ------------------------------------------
        # Logging
        # ------------------------------------------
        history["total"].append(total.item())
        history["pde"].append(loss_pde.item())
        history["norm"].append(loss_norm.item())
        history["bc"].append(loss_bc.item())

        if epoch % 500 == 0:
            print(
                f"[{epoch}] "
                f"total={total.item():.3e} | "
                f"PDE={loss_pde.item():.3e} | "
                f"norm={loss_norm.item():.3e} | "
                f"BC={loss_bc.item():.3e}"
            )
    return history

def free_particle(N=4096):
    x, y = sample_cell(N, a1, a2)
    kx, ky = sample_IBZ(N, radius=0)
    uR = torch.ones(N, 1, device=device)
    uI = torch.zeros(N, 1, device=device)
    E  = 0.5*(kx**2 + ky**2)

    loss = bloch_pde_loss(uR, uI, E, x, y, kx, ky, lambda x,y: 0.0*x)
    print("free particle PDE loss:", loss.item())

if __name__ == "__main__":
    print("\n Starting Bloch PINN training with V0=10 (transfer learning)...\n")
    print(f"Using device: {device}")

    # Load pretrained model from V0=1
    model = BlochPINN(B=1).to(device)
    model.load_state_dict(torch.load("results/bloch_model_20251215_003720.pt"))
    print("Loaded pretrained model from V0=1")

    os.makedirs("results", exist_ok=True)

    # Train with lower learning rate for fine-tuning
    history = train_bloch_pinn(
        model,
        honeycomb_potential_v10,  # V0=10 now!
        a1, a2,
        n_epochs=10000,           # Might need fewer epochs
        lr=5e-4,                  # Lower LR for fine-tuning
        rebalance=False
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"results/bloch_model_V10_{timestamp}.pt"
    loss_path  = f"results/loss_history_V10_{timestamp}.npy"

    torch.save(model.state_dict(), model_path)
    np.save(loss_path, history)

    print(f"\nTraining finished — model saved to:\n  {model_path}\n")
    
    # print("\n Starting Bloch PINN training...\n")
    # print(f"Using device: {device}")

    # # init model
    # model = BlochPINN(B=1).to(device)
    # os.makedirs("results", exist_ok=True)

    # profile_training_speed()
    # # train
    # history = train_bloch_pinn(
    #     model,
    #     honeycomb_potential,
    #     a1, a2,
    #     n_epochs=20000
    # )
    
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # model_path = f"results/bloch_model_{timestamp}.pt"
    # loss_path  = f"results/loss_history_{timestamp}.npy"

    # torch.save(model.state_dict(), model_path)
    # np.save(loss_path, history)

    # print(f"\nTraining finished — model saved to:\n  {model_path}\n")