"""
Latent Diffusion Language Model:  the LDM-faithful hybrid.

Design (matches the architecture we discussed):
  Stage 1 :  a KL-regularized autoencoder over TOKENS.
             * bottom-up encoder  : tokens (L) -> ONE coarse latent per patch  (C = L/P latents)
             * top-down decoder    : coarse latents (C) -> reconstructed tokens (L)
             This is PHOTON's compress/decompress skeleton, made *variational*
             (light KL) so the latent space is smooth & ~standardized, which is
             what a continuous diffusion prior needs as a target.
  Stage 2 :  a continuous Gaussian DDPM over the COARSE LATENT STREAM z in R^{C x d}.
             The encoder/decoder are FROZEN (the LDM two-stage recipe).
             A small bidirectional transformer epsilon_theta(z_t, t) denoises the latents.
  Sample  :  z ~ N(0,I) --DDPM reverse--> z0  --frozen top-down decoder--> tokens.

Scope note (honest): the latent diffusion here is *fully bidirectional* over the C
latents — the simplest faithful core of (B). To recover PHOTON's KV-cache win you
make the coarse level BLOCK-causal (AR across coarse blocks, diffusion within a
block); the hook for that is marked `# [block-causal hook]` in the denoiser.

Smoke test: a synthetic hierarchical grammar (fixed 4-token chunk archetypes +
a sparse Markov chain over chunk types). We check that (1) the AE reconstructs,
(2) the diffusion loss drops, and (3) UNCONDITIONAL samples reproduce the local
grammar (valid chunks) and global grammar (legal chunk transitions) far above
the random baseline.
"""
# Rerecence 
# PHOTON https://arxiv.org/abs/2512.20687
# BD3-LM https://arxiv.org/abs/2506.13759
# latent diffusion this code

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEVICE = "cpu"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class Cfg:
    vocab       = 32       # token vocabulary
    seq_len     = 64       # L
    patch       = 4        # P  (tokens per chunk)
    n_latent    = seq_len // patch      # C = 16 coarse latents
    latent_dim  = 16       # d  per latent
    d_model     = 64
    n_heads     = 4
    ae_layers   = 2
    dn_layers   = 4
    # diffusion
    T           = 200
    coarse_block = 4       # latents per coarse block (block-causal variant): B_c = C/coarse_block
    # grammar
    n_archetype = 8        # number of distinct 4-token chunk patterns
    kl_weight   = 1e-3     # light KL, LDM-style
    # train
    batch       = 48
    ae_steps    = 300
    dn_steps    = 2600
    lr          = 2e-3

C = Cfg()


# --------------------------------------------------------------------------- #
# Toy hierarchical grammar  (verifiable local + global structure)
# --------------------------------------------------------------------------- #
class Grammar:
    """chunk archetypes (local structure) + Markov chain over types (global)."""
    def __init__(self, cfg: Cfg, seed=1):
        g = torch.Generator().manual_seed(seed)
        # each archetype is a fixed pattern of `patch` tokens
        self.archetypes = torch.randint(0, cfg.vocab, (cfg.n_archetype, cfg.patch), generator=g)
        # sparse Markov transition: each type -> ~3 legal successors
        T = torch.zeros(cfg.n_archetype, cfg.n_archetype)
        for i in range(cfg.n_archetype):
            succ = torch.randperm(cfg.n_archetype, generator=g)[:3]
            T[i, succ] = 1.0
        self.trans = T / T.sum(1, keepdim=True)
        self.cfg = cfg

    def sample(self, n):
        cfg = self.cfg
        types = torch.empty(n, cfg.n_latent, dtype=torch.long)
        t = torch.randint(0, cfg.n_archetype, (n,))          # initial chunk type per seq
        for c in range(cfg.n_latent):                        # C batched steps (not n*C)
            types[:, c] = t
            t = torch.multinomial(self.trans[t], 1).squeeze(1)
        toks = self.archetypes[types]                        # (n, C, P) fully vectorized
        return toks.reshape(n, cfg.seq_len)

    # ---- evaluation helpers ----
    def classify_chunks(self, x):
        """map each patch to its archetype id, or -1 if it matches none exactly."""
        n = x.shape[0]
        chunks = x.view(n, self.cfg.n_latent, self.cfg.patch)          # (n,C,P)
        # exact match against archetypes
        eq = (chunks[:, :, None, :] == self.archetypes[None, None]).all(-1)  # (n,C,A)
        ids = torch.where(eq.any(-1), eq.float().argmax(-1), torch.full_like(eq[..., 0], -1, dtype=torch.long))
        return ids  # (n, C), -1 = invalid chunk

    def score(self, x):
        ids = self.classify_chunks(x)
        valid = (ids >= 0)
        local = valid.float().mean().item()  # fraction of chunks that are legal archetypes
        # global: among adjacent valid pairs, fraction that are legal transitions
        legal_mask = self.trans > 0
        pair_ok, pair_tot = 0, 0
        for b in range(ids.shape[0]):
            for c in range(self.cfg.n_latent - 1):
                a, d = ids[b, c].item(), ids[b, c+1].item()
                if a >= 0 and d >= 0:
                    pair_tot += 1
                    pair_ok += int(legal_mask[a, d])
        global_ = (pair_ok / pair_tot) if pair_tot else 0.0
        return local, global_


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def pos_emb(n, d):
    pe = torch.zeros(n, d)
    pos = torch.arange(n).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))

    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        x = x + self.att(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x


# --------------------------------------------------------------------------- #
# Stage 1 : variational bottom-up / top-down autoencoder
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    """tokens (B,L) -> per-patch latent distribution (mu, logvar) of shape (B,C,d)."""
    def __init__(self, cfg):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.register_buffer("pe", pos_emb(cfg.seq_len, cfg.d_model))
        self.blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads) for _ in range(cfg.ae_layers)])
        self.to_lat = nn.Linear(cfg.d_model * cfg.patch, 2 * cfg.latent_dim)
        self.cfg = cfg

    def forward(self, x):
        h = self.emb(x) + self.pe
        for blk in self.blocks:
            h = blk(h)
        # patchify (bottom-up compression: P token states -> 1 latent)
        h = h.view(x.shape[0], self.cfg.n_latent, self.cfg.patch * self.cfg.d_model)
        mu, logvar = self.to_lat(h).chunk(2, dim=-1)
        return mu, logvar


class Decoder(nn.Module):
    """coarse latents (B,C,d) -> token logits (B,L,V) via top-down expansion."""
    def __init__(self, cfg):
        super().__init__()
        self.up = nn.Linear(cfg.latent_dim, cfg.d_model * cfg.patch)
        self.register_buffer("pe", pos_emb(cfg.seq_len, cfg.d_model))
        self.blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads) for _ in range(cfg.ae_layers)])
        self.head = nn.Linear(cfg.d_model, cfg.vocab)
        self.cfg = cfg

    def forward(self, z):
        B = z.shape[0]
        # top-down: each coarse latent expands to its P token slots
        h = self.up(z).view(B, self.cfg.seq_len, self.cfg.d_model) + self.pe
        for blk in self.blocks:                # (full attention here; swap to local
            h = blk(h)                          #  windows to get PHOTON's decode locality)
        return self.head(h)


class LatentAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.enc, self.dec, self.cfg = Encoder(cfg), Decoder(cfg), cfg

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        logits = self.dec(z)
        recon = F.cross_entropy(logits.reshape(-1, self.cfg.vocab), x.reshape(-1))
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()
        return recon, kl, logits


# --------------------------------------------------------------------------- #
# Stage 2 : continuous Gaussian DDPM over the coarse latent stream
# --------------------------------------------------------------------------- #
class Diffusion:
    """Continuous Gaussian DDPM with x0-parameterization.

    We predict z0 (not epsilon): epsilon-prediction up-weights high-frequency detail
    (here, per-chunk archetype identity), which made local structure perfect but left
    the low-frequency inter-latent correlations — the GLOBAL grammar — underfit.
    x0-prediction plus the true posterior sampler recovers that global structure.
    """
    def __init__(self, T):
        betas = torch.linspace(1e-4, 0.02, T)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.acp = torch.cumprod(self.alphas, 0)            # \bar\alpha_t
        self.acp_prev = torch.cat([torch.ones(1), self.acp[:-1]])
        # q(x_{t-1} | x_t, x0) posterior coefficients
        self.post_var = betas * (1 - self.acp_prev) / (1 - self.acp)
        self.c_x0 = betas * self.acp_prev.sqrt() / (1 - self.acp)
        self.c_xt = (1 - self.acp_prev) * self.alphas.sqrt() / (1 - self.acp)
        self.T = T

    def q_sample(self, x0, t, noise):
        a = self.acp[t].sqrt().view(-1, 1, 1)
        b = (1 - self.acp[t]).sqrt().view(-1, 1, 1)
        return a * x0 + b * noise

    @torch.no_grad()
    def sample(self, model, shape):
        x = torch.randn(shape)
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, dtype=torch.long)
            x0_hat = model(x, t)                            # model predicts z0
            mean = self.c_x0[i] * x0_hat + self.c_xt[i] * x
            if i > 0:
                x = mean + self.post_var[i].clamp(min=1e-20).sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x


class Denoiser(nn.Module):
    """x0_theta(z_t, t): bidirectional transformer over the C latents; predicts z0."""
    def __init__(self, cfg):
        super().__init__()
        self.inp = nn.Linear(cfg.latent_dim, cfg.d_model)
        self.register_buffer("pe", pos_emb(cfg.n_latent, cfg.d_model))
        self.t_mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.SiLU(),
                                   nn.Linear(cfg.d_model, cfg.d_model))
        self.blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads) for _ in range(cfg.dn_layers)])
        self.out = nn.Linear(cfg.d_model, cfg.latent_dim)
        self.cfg = cfg

    def t_embed(self, t):
        d = self.cfg.d_model
        half = d // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / half)
        a = t.float()[:, None] * freqs[None]
        return self.t_mlp(torch.cat([a.sin(), a.cos()], -1))

    def forward(self, z, t):
        h = self.inp(z) + self.pe + self.t_embed(t)[:, None, :]
        mask = None  # [block-causal hook] set a block-causal mask over the C latents
        for blk in self.blocks:                # to trade full bidirectionality for
            h = blk(h, attn_mask=mask)          # KV-cacheable coarse AR-across-blocks.
        return self.out(h)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_stage1(ae, gram, cfg):
    opt = torch.optim.AdamW(ae.parameters(), lr=cfg.lr)
    for step in range(cfg.ae_steps):
        x = gram.sample(cfg.batch)
        recon, kl, _ = ae(x)
        loss = recon + cfg.kl_weight * kl
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0 or step == cfg.ae_steps - 1:
            with torch.no_grad():
                acc = (ae(x)[2].argmax(-1) == x).float().mean().item()
            print(f"  [AE {step:4d}] recon={recon.item():.3f} kl={kl.item():.3f} acc={acc:.3f}")
    return ae


def compute_scale(ae, gram, cfg):
    """LDM-style latent scale: normalize latents to ~unit std before diffusion."""
    with torch.no_grad():
        mu, _ = ae.enc(gram.sample(1024))
    return 1.0 / mu.std().item()


def encode_pool(ae, gram, scale, cfg, pool=16384):
    """Cache the frozen encoder's latents once; stage-2 then resamples from this
    pool so each step is denoiser-only (the AE is frozen, so its outputs are fixed)."""
    with torch.no_grad():
        zs = []
        for _ in range(0, pool, 512):
            mu, _ = ae.enc(gram.sample(512))
            zs.append(mu * scale)
    return torch.cat(zs, 0)


def train_stage2(denoiser, ae, diff, gram, scale, cfg):
    for p in ae.parameters():
        p.requires_grad_(False)          # freeze AE (two-stage recipe)
    z_pool = encode_pool(ae, gram, scale, cfg)
    opt = torch.optim.AdamW(denoiser.parameters(), lr=cfg.lr)
    for step in range(cfg.dn_steps):
        idx = torch.randint(0, z_pool.shape[0], (cfg.batch,))
        z0 = z_pool[idx]
        t = torch.randint(0, cfg.T, (cfg.batch,))
        noise = torch.randn_like(z0)
        zt = diff.q_sample(z0, t, noise)
        pred = denoiser(zt, t)               # predicts z0 (x0-parameterization)
        loss = F.mse_loss(pred, z0)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0 or step == cfg.dn_steps - 1:
            print(f"  [DN {step:4d}] mse={loss.item():.4f}")
    return denoiser


@torch.no_grad()
def generate(denoiser, ae, diff, scale, cfg, n=256):
    z = diff.sample(denoiser, (n, cfg.n_latent, cfg.latent_dim)) / scale
    logits = ae.dec(z)
    return logits.argmax(-1)


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
def main():
    print("=" * 64)
    print("Latent Diffusion LM  (Option B)  —  smoke test")
    print(f"  L={C.seq_len}  P={C.patch}  C={C.n_latent}  latent_dim={C.latent_dim}"
          f"  |  compression {C.seq_len}->{C.n_latent} tokens/latents")
    print("=" * 64)

    gram = Grammar(C)

    # sanity: real data scores ~1.0 on both axes; random baseline is ~0
    real = gram.sample(256)
    rnd  = torch.randint(0, C.vocab, (256, C.seq_len))
    rl, rg = gram.score(real)
    zl, zg = gram.score(rnd)
    print(f"[grammar] real data   local={rl:.3f} global={rg:.3f}")
    print(f"[grammar] random toks local={zl:.4f} global={zg:.4f}  (baseline)")

    print("\n[Stage 1] variational bottom-up/top-down autoencoder")
    ae = LatentAE(C)
    train_stage1(ae, gram, C)
    with torch.no_grad():
        x = gram.sample(512)
        ae_acc = (ae(x)[2].argmax(-1) == x).float().mean().item()
    print(f"  -> reconstruction token accuracy = {ae_acc:.3f}")

    scale = compute_scale(ae, gram, C)
    print(f"  -> latent scale factor = {scale:.3f}")

    print("\n[Stage 2] continuous DDPM over coarse latent stream")
    diff = Diffusion(C.T)
    denoiser = Denoiser(C)
    train_stage2(denoiser, ae, diff, gram, scale, C)

    print("\n[Sampling] z~N(0,I) -> reverse diffusion -> top-down decode")
    samp = generate(denoiser, ae, diff, scale, C, n=256)
    sl, sg = gram.score(samp)
    print(f"  -> generated  local={sl:.3f}  global={sg:.3f}")
    print(f"     (vs random baseline local={zl:.4f} global={zg:.4f})")

    # example decoded sequence, shown as chunk-type ids (-1 = invalid chunk)
    ids = gram.classify_chunks(samp[:1])[0].tolist()
    print(f"  -> sample #0 chunk-type ids: {ids}")

    print("\n" + "=" * 64)
    checks = {
        "AE reconstructs (acc > 0.90)":            ae_acc > 0.90,
        "diffusion learns local grammar (>0.50)":  sl > 0.50,
        "local >> random baseline (10x)":          sl > max(10 * zl, 0.05),
        "global transitions plausible (>0.60)":    sg > 0.60,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    allok = all(checks.values())
    print("=" * 64)
    print("SMOKE TEST:", "PASS ✅" if allok else "FAIL ❌")
    return allok


# --------------------------------------------------------------------------- #
# Block-causal (semi-AR) variant  —  the KV-cacheable option (B)
# --------------------------------------------------------------------------- #
# Group the C coarse latents into B_c blocks. AR ACROSS blocks (a block sees only
# clean earlier blocks) + diffusion WITHIN a block (bidirectional). This is BD3-LM
# applied at the LATENT level: it restores KV caching (earlier blocks' K/V are fixed
# and cacheable across the current block's denoising steps and across blocks), which
# the fully-bidirectional core above gives up.
def block_causal_mask(n_latent, block):
    """additive attention mask (C,C): 0 if j in same-or-earlier block than i, else -inf."""
    blk = torch.arange(n_latent) // block
    allow = blk[None, :] <= blk[:, None]        # i attends j iff block(j) <= block(i)
    return torch.where(allow, 0.0, float("-inf"))


class BlockCausalDenoiser(nn.Module):
    """x0_theta over C latents with a block-causal mask; predicts z0 for all positions.
    Earlier blocks are fed CLEAN, the active block is fed noised at timestep t."""
    def __init__(self, cfg):
        super().__init__()
        self.inp = nn.Linear(cfg.latent_dim, cfg.d_model)
        self.register_buffer("pe", pos_emb(cfg.n_latent, cfg.d_model))
        self.t_mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.SiLU(),
                                   nn.Linear(cfg.d_model, cfg.d_model))
        self.blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads) for _ in range(cfg.dn_layers)])
        self.out = nn.Linear(cfg.d_model, cfg.latent_dim)
        self.register_buffer("mask", block_causal_mask(cfg.n_latent, cfg.coarse_block))
        self.cfg = cfg

    def t_embed(self, t):
        half = self.cfg.d_model // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / half)
        a = t.float()[:, None] * freqs[None]
        return self.t_mlp(torch.cat([a.sin(), a.cos()], -1))

    def forward(self, z, t):
        h = self.inp(z) + self.pe + self.t_embed(t)[:, None, :]
        for blk in self.blocks:
            h = blk(h, attn_mask=self.mask)
        return self.out(h)


def train_stage2_bc(denoiser, ae, diff, gram, scale, cfg):
    """Semi-AR training: per example sample an active block b and a timestep t;
    noise block b, keep earlier blocks clean; loss on block b only. Unbiased
    estimator of sum_b E_t [ ||z0^b - x0_theta(z_t^b, z0^{<b})||^2 ]."""
    for p in ae.parameters():
        p.requires_grad_(False)
    z_pool = encode_pool(ae, gram, scale, cfg)
    B_c = cfg.n_latent // cfg.coarse_block
    blk_id = torch.arange(cfg.n_latent) // cfg.coarse_block         # (C,)
    opt = torch.optim.AdamW(denoiser.parameters(), lr=cfg.lr)
    for step in range(cfg.dn_steps):
        idx = torch.randint(0, z_pool.shape[0], (cfg.batch,))
        z0 = z_pool[idx]                                            # (B,C,d) clean
        b = torch.randint(0, B_c, (cfg.batch,))                    # active block per example
        t = torch.randint(0, cfg.T, (cfg.batch,))
        active = (blk_id[None, :] == b[:, None])                   # (B,C) bool, active-block positions
        noise = torch.randn_like(z0)
        zt = diff.q_sample(z0, t, noise)
        zin = torch.where(active[..., None], zt, z0)               # noise active block, clean elsewhere
        pred = denoiser(zin, t)
        loss = (((pred - z0) ** 2) * active[..., None]).sum() / active.sum() / z0.shape[-1]
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0 or step == cfg.dn_steps - 1:
            print(f"  [BC {step:4d}] mse={loss.item():.4f}", flush=True)
    return denoiser


@torch.no_grad()
def generate_bc(denoiser, ae, diff, scale, cfg, n=256):
    """Block-by-block generation: diffuse each coarse block conditioned on the clean
    previously-generated blocks (KV-cacheable across blocks in a real impl)."""
    B_c = cfg.n_latent // cfg.coarse_block
    blk_id = torch.arange(cfg.n_latent) // cfg.coarse_block
    z = torch.zeros(n, cfg.n_latent, cfg.latent_dim)               # filled block by block
    for b in range(B_c):
        active = (blk_id == b)
        x = torch.randn(n, cfg.n_latent, cfg.latent_dim)           # noise for active block
        x = torch.where(active[None, :, None], x, z)               # clean context in earlier blocks
        for i in reversed(range(cfg.T)):
            t = torch.full((n,), i, dtype=torch.long)
            x0_hat = denoiser(x, t)
            mean = diff.c_x0[i] * x0_hat + diff.c_xt[i] * x
            if i > 0:
                step = mean + diff.post_var[i].clamp(min=1e-20).sqrt() * torch.randn_like(x)
            else:
                step = mean
            x = torch.where(active[None, :, None], step, z)        # keep context frozen
        z = torch.where(active[None, :, None], x, z)               # commit generated block
    return ae.dec(z / scale).argmax(-1)


def main_bc():
    print("=" * 64)
    print("Latent Diffusion LM  (Option B, BLOCK-CAUSAL / semi-AR)  —  smoke test")
    print(f"  C={C.n_latent} latents in B_c={C.n_latent // C.coarse_block} coarse blocks "
          f"of {C.coarse_block}  |  AR across blocks + diffusion within")
    print("=" * 64)
    gram = Grammar(C)
    rnd = torch.randint(0, C.vocab, (256, C.seq_len)); zl, zg = gram.score(rnd)

    print("\n[Stage 1] reuse/train variational autoencoder")
    ae = LatentAE(C)
    train_stage1(ae, gram, C)
    with torch.no_grad():
        x = gram.sample(512); ae_acc = (ae(x)[2].argmax(-1) == x).float().mean().item()
    scale = compute_scale(ae, gram, C)
    print(f"  -> recon acc={ae_acc:.3f}  scale={scale:.3f}")

    print("\n[Stage 2] block-causal latent diffusion (x0-pred)")
    diff = Diffusion(C.T)
    dn = BlockCausalDenoiser(C)
    train_stage2_bc(dn, ae, diff, gram, scale, C)

    print("\n[Sampling] block-by-block diffusion -> top-down decode")
    samp = generate_bc(dn, ae, diff, scale, C, n=256)
    sl, sg = gram.score(samp)
    print(f"  -> generated  local={sl:.3f}  global={sg:.3f}   (real=1.0, rand-valid global~0.375)")
    print(f"  -> sample #0 chunk-type ids: {gram.classify_chunks(samp[:1])[0].tolist()}")

    print("\n" + "=" * 64)
    checks = {
        "AE reconstructs (acc > 0.90)":           ae_acc > 0.90,
        "learns local grammar (>0.50)":           sl > 0.50,
        "local >> random baseline (10x)":         sl > max(10 * zl, 0.05),
        "global transitions plausible (>0.60)":   sg > 0.60,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    allok = all(checks.values())
    print("=" * 64)
    print("SMOKE TEST:", "PASS ✅" if allok else "FAIL ❌")
    return allok


if __name__ == "__main__":
    import sys
    entry = main_bc if (len(sys.argv) > 1 and sys.argv[1] == "block_causal") else main
    raise SystemExit(0 if entry() else 1)


