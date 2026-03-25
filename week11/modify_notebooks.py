#!/usr/bin/env python3
"""Modify exercises.ipynb and solutions.ipynb to replace flow matching with VP-SDE/PFODE on MNIST."""

import json
import copy

# ── Cell sources ──────────────────────────────────────────────────────────────

HEADER_SOURCE = [
    "# Week 11: Continuous-Time Diffusion — VP-SDE and Probability Flow ODE\n",
    "\n",
    "In Weeks 9 and 10 we worked with **discrete** timesteps $t \\in \\{0, 1, \\ldots, T-1\\}$.  \n",
    "This week we ask: *what happens as $T \\to \\infty$?*\n",
    "\n",
    "We will:\n",
    "1. Reformulate discrete DDPM as a **continuous-time VP-SDE** with $t \\in [0, 1]$\n",
    "2. Show that the same score model can be sampled with **any number of ODE steps** at inference time\n",
    "3. Scale to MNIST images (digits 0 and 1) using the same VP-SDE framework\n",
    "\n",
    "**Datasets used:**\n",
    "- Cat 2D point cloud (from `cat.png`, same as Week 5)\n",
    "- MNIST digits 0 and 1\n",
    "\n",
    "`# TODO` marks exercises for you to complete.",
]

PART2_DISCUSSION_SOURCE = [
    "### Discussion\n",
    "\n",
    "1. At what value of $N$ does the cat shape first become recognisable? At what $N$ does it look sharp?\n",
    "2. The ODE sampler is **deterministic** — for a fixed initial noise, it always produces the same output. What does this imply about diversity compared to stochastic DDPM?\n",
    "3. Why might VP-SDE probability flow ODE paths require more steps than you might expect? Think about the geometry of the trajectories.",
]

FINAL_DISCUSSION_SOURCE = [
    "### Final Discussion\n",
    "\n",
    "1. Compare the minimum viable $N$ for the cat 2D experiment (Part 2) vs MNIST (Part 3). Why do images require more steps for recognisable quality?\n",
    "2. In Part 3 we trained a single unconditional model on digits 0 *and* 1 mixed together. What would you expect to see in the samples — pure 0s, pure 1s, or a mix? Does your observation match the expectation?\n",
    "3. **Connecting Weeks 9–11:**  \n",
    "   - Week 9: discrete DDPM, score matching  \n",
    "   - Week 10: conditioning via classifier guidance and CFG  \n",
    "   - Week 11: continuous time, VP-SDE, probability flow ODE  \n",
    "   \n",
    "   What is the single most important idea connecting all three frameworks?",
]

PART3_HEADER_NEW_SOURCE = [
    "---\n",
    "## Part 3: VP-SDE and PFODE on MNIST (Digits 0 and 1)\n",
    "\n",
    "We now apply the **same** VP-SDE framework from Parts 1 and 2 to **28×28 grayscale images**, using a small U-Net as the noise-prediction network.\n",
    "\n",
    "The training objective is identical to Part 2 — minimise $\\|\\varepsilon_\\theta(x_t, t) - \\varepsilon\\|^2$ with continuous $t \\sim \\mathcal{U}[0,1]$.  \n",
    "The PFODE sampler uses the exact same DDIM-style update as before.",
]

UNET_IMPORT_NEW_SOURCE = [
    "from unet_flow import MiniUNetContinuous\n",
    "\n",
    "unet = MiniUNetContinuous(in_channels=1, time_emb_dim=64).to(device)\n",
    "print(f'MiniUNetContinuous parameters: {sum(p.numel() for p in unet.parameters()):,}')",
]

MNIST_UNET_TRAINING_EXERCISES_SOURCE = [
    "# ── VP-SDE training on MNIST images ─────────────────────────────────────────\n",
    "opt_unet = torch.optim.Adam(unet.parameters(), lr=1e-3)\n",
    "epochs_img = 20\n",
    "losses_img = []\n",
    "\n",
    "for epoch in range(epochs_img):\n",
    "    for x_0, _ in img_loader:\n",
    "        x_0 = x_0.to(device)\n",
    "        B = x_0.shape[0]\n",
    "\n",
    "        # TODO: sample t ~ Uniform[0, 1]\n",
    "        t = ...\n",
    "\n",
    "        # TODO: corrupt x_0 using q_sample (works for 4D tensors — check q_sample's broadcast logic)\n",
    "        x_t, noise = ...\n",
    "\n",
    "        eps_pred = unet(x_t, t)\n",
    "        loss = F.mse_loss(eps_pred, noise)\n",
    "\n",
    "        opt_unet.zero_grad(); loss.backward(); opt_unet.step()\n",
    "        losses_img.append(loss.item())\n",
    "\n",
    "    print(f'Epoch {epoch+1:2d}/{epochs_img} | loss {np.mean(losses_img[-len(img_loader):]):.4f}')\n",
    "\n",
    "plt.figure(figsize=(7, 3))\n",
    "plt.plot(losses_img); plt.xlabel('batch'); plt.ylabel('MSE loss')\n",
    "plt.title('VP-SDE training on MNIST')\n",
    "plt.tight_layout(); plt.show()",
]

MNIST_UNET_TRAINING_SOLUTIONS_SOURCE = [
    "# ── VP-SDE training on MNIST images ─────────────────────────────────────────\n",
    "opt_unet = torch.optim.Adam(unet.parameters(), lr=1e-3)\n",
    "epochs_img = 20\n",
    "losses_img = []\n",
    "\n",
    "for epoch in range(epochs_img):\n",
    "    for x_0, _ in img_loader:\n",
    "        x_0 = x_0.to(device)\n",
    "        B = x_0.shape[0]\n",
    "\n",
    "        # Sample t ~ Uniform[0, 1]\n",
    "        t = torch.rand(B, device=device)\n",
    "\n",
    "        # Corrupt x_0 — q_sample broadcasts over spatial dims automatically\n",
    "        x_t, noise = q_sample(x_0, t)\n",
    "\n",
    "        eps_pred = unet(x_t, t)\n",
    "        loss = F.mse_loss(eps_pred, noise)\n",
    "\n",
    "        opt_unet.zero_grad(); loss.backward(); opt_unet.step()\n",
    "        losses_img.append(loss.item())\n",
    "\n",
    "    print(f'Epoch {epoch+1:2d}/{epochs_img} | loss {np.mean(losses_img[-len(img_loader):]):.4f}')\n",
    "\n",
    "plt.figure(figsize=(7, 3))\n",
    "plt.plot(losses_img); plt.xlabel('batch'); plt.ylabel('MSE loss')\n",
    "plt.title('VP-SDE training on MNIST')\n",
    "plt.tight_layout(); plt.show()",
]

MNIST_PFODE_SAMPLER_EXERCISES_SOURCE = [
    "@torch.no_grad()\n",
    "def sample_pfode_mnist(model, n_samples=8, N=100):\n",
    '    """\n',
    "    PFODE sampler for images — same DDIM update as sample_ode() in Part 2.\n",
    "    Integrates from t=1 (noise) to t≈0 (data) in N steps.\n",
    '    """\n',
    "    model.eval()\n",
    "    x = torch.randn(n_samples, 1, 28, 28, device=device)\n",
    "    ts = torch.linspace(1.0, 1e-3, N + 1, device=device)\n",
    "\n",
    "    for i in range(N):\n",
    "        t_curr = ts[i]\n",
    "        t_next = ts[i + 1]\n",
    "\n",
    "        eps_pred = model(x, t_curr.expand(n_samples))\n",
    "\n",
    "        a_c = alpha(t_curr)\n",
    "        a_n = alpha(t_next)\n",
    "        s_c = sigma(t_curr)\n",
    "        s_n = sigma(t_next)\n",
    "\n",
    "        # TODO: DDIM update step — identical formula to Part 2's sample_ode()\n",
    "        x = ...\n",
    "\n",
    "    return x.cpu()",
]

MNIST_PFODE_SAMPLER_SOLUTIONS_SOURCE = [
    "@torch.no_grad()\n",
    "def sample_pfode_mnist(model, n_samples=8, N=100):\n",
    '    """\n',
    "    PFODE sampler for images — same DDIM update as sample_ode() in Part 2.\n",
    "    Integrates from t=1 (noise) to t≈0 (data) in N steps.\n",
    '    """\n',
    "    model.eval()\n",
    "    x = torch.randn(n_samples, 1, 28, 28, device=device)\n",
    "    ts = torch.linspace(1.0, 1e-3, N + 1, device=device)\n",
    "\n",
    "    for i in range(N):\n",
    "        t_curr = ts[i]\n",
    "        t_next = ts[i + 1]\n",
    "\n",
    "        eps_pred = model(x, t_curr.expand(n_samples))\n",
    "\n",
    "        a_c = alpha(t_curr)\n",
    "        a_n = alpha(t_next)\n",
    "        s_c = sigma(t_curr)\n",
    "        s_n = sigma(t_next)\n",
    "\n",
    "        # DDIM update — identical to Part 2\n",
    "        x = (a_n / a_c) * x + (s_n - (a_n / a_c) * s_c) * eps_pred\n",
    "\n",
    "    return x.cpu()",
]

MNIST_PFODE_SWEEP_SOURCE = [
    "# ── Explore: image quality vs number of PFODE steps ─────────────────────────\n",
    "step_counts = [5, 10, 20, 50, 100, 500]\n",
    "n_per_row   = 8\n",
    "\n",
    "fig, axes = plt.subplots(len(step_counts), n_per_row, figsize=(16, 2.5 * len(step_counts)))\n",
    "\n",
    "for row, N in enumerate(step_counts):\n",
    "    imgs = sample_pfode_mnist(unet, n_samples=n_per_row, N=N)\n",
    "    for col in range(n_per_row):\n",
    "        img = (imgs[col].squeeze().numpy() + 1) / 2   # [-1,1] -> [0,1]\n",
    "        img = np.clip(img, 0, 1)\n",
    "        axes[row, col].imshow(img, cmap='gray', vmin=0, vmax=1)\n",
    "        axes[row, col].axis('off')\n",
    "    axes[row, 0].set_ylabel(f'N = {N}', fontsize=11)\n",
    "\n",
    "plt.suptitle('VP-SDE PFODE on MNIST: sample quality vs steps', y=1.01)\n",
    "plt.tight_layout(); plt.show()",
]

# ── IDs to delete ─────────────────────────────────────────────────────────────
IDS_TO_DELETE = {
    'part3-header', 'flow-utils', 'flow-mlp', 'flow-training', 'flow-sampler',
    'flow-sweep-comparison', 'part3-discussion', 'part4-header', 'unet-import',
    'mnist-training', 'mnist-sampler',
}

# Also delete the old mnist-sweep cell (it refs sample_flow_mnist)
IDS_TO_DELETE.add('mnist-sweep')


def make_code_cell(cell_id, source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def make_markdown_cell(cell_id, source):
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source,
    }


def modify_notebook(nb, is_exercises: bool):
    cells = nb['cells']

    # 1. Update header cell
    for c in cells:
        if c.get('id') == 'header':
            c['source'] = HEADER_SOURCE

    # 2. Update part2-discussion cell
    for c in cells:
        if c.get('id') == 'part2-discussion':
            c['source'] = PART2_DISCUSSION_SOURCE

    # 3. Update final-discussion cell
    for c in cells:
        if c.get('id') == 'final-discussion':
            c['source'] = FINAL_DISCUSSION_SOURCE

    # 4. Delete unwanted cells
    cells = [c for c in cells if c.get('id') not in IDS_TO_DELETE]

    # 5. Find index of mnist-data cell and insert new cells after it
    mnist_data_idx = None
    for i, c in enumerate(cells):
        if c.get('id') == 'mnist-data':
            mnist_data_idx = i
            break

    if mnist_data_idx is None:
        raise ValueError("mnist-data cell not found!")

    # Build the 5 new cells
    new_cells = [
        make_markdown_cell('part3-header-new', PART3_HEADER_NEW_SOURCE),
        make_code_cell('unet-import-new', UNET_IMPORT_NEW_SOURCE),
        make_code_cell(
            'mnist-unet-training',
            MNIST_UNET_TRAINING_EXERCISES_SOURCE if is_exercises else MNIST_UNET_TRAINING_SOLUTIONS_SOURCE,
        ),
        make_code_cell(
            'mnist-pfode-sampler',
            MNIST_PFODE_SAMPLER_EXERCISES_SOURCE if is_exercises else MNIST_PFODE_SAMPLER_SOLUTIONS_SOURCE,
        ),
        make_code_cell('mnist-pfode-sweep', MNIST_PFODE_SWEEP_SOURCE),
    ]

    # Insert after mnist-data
    insert_at = mnist_data_idx + 1
    cells = cells[:insert_at] + new_cells + cells[insert_at:]

    nb['cells'] = cells
    return nb


# ── Process both notebooks ────────────────────────────────────────────────────

for filename, is_exercises in [('exercises.ipynb', True), ('solutions.ipynb', False)]:
    path = f'/Users/rajit906/Downloads/ATML2026_DGM/week11/{filename}'
    with open(path) as f:
        nb = json.load(f)

    nb = modify_notebook(nb, is_exercises)

    with open(path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')

    print(f'Written: {filename}  ({len(nb["cells"])} cells)')

# ── Verification: check no forbidden strings remain in cell sources ───────────
FORBIDDEN = ['flow_interpolate', 'flow_target', 'FlowMLP', 'MiniUNetFlow',
             'sample_flow', 'Flow Matching']

for filename in ['exercises.ipynb', 'solutions.ipynb']:
    path = f'/Users/rajit906/Downloads/ATML2026_DGM/week11/{filename}'
    with open(path) as f:
        nb = json.load(f)
    issues = []
    for c in nb['cells']:
        src = ''.join(c['source'])
        for term in FORBIDDEN:
            if term in src:
                issues.append(f"  Cell {c.get('id')} contains '{term}'")
    if issues:
        print(f'\nFORBIDDEN STRINGS in {filename}:')
        for issue in issues:
            print(issue)
    else:
        print(f'OK: no forbidden strings in {filename}')
