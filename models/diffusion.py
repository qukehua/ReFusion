"""Forward noising and deterministic DDIM for future-only motion DCT latents."""

import math
import numpy as np
import torch


def sqrt_beta_schedule(timesteps, s=0.0001):
    t = torch.linspace(0, timesteps, timesteps + 1) / timesteps
    alphas_cumprod = 1 - torch.sqrt(t + s)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule from Improved DDPM (Nichol and Dhariwal)."""
    t = torch.linspace(0, timesteps, timesteps + 1) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3., end=3., tau=0.7, clamp_min=1e-5):
    """Sigmoid schedule from https://arxiv.org/abs/2212.11972, Figure 8."""
    t = torch.linspace(0, timesteps, timesteps + 1) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, clamp_min, 0.999)


def logarithmic_beta_schedule(timesteps, start=1e-4, end=2e-2):
    """Interpolate beta values uniformly in log-space."""
    betas = torch.logspace(math.log10(start), math.log10(end), timesteps)
    return torch.clip(betas, 0, 0.999)


class Diffusion:
    """Noise-prediction diffusion of future coefficients [B,M,3J].

    Time indices are 0,...,K: alpha_hat[0] = 1 and beta[k] is the variance
    of forward transition k, for 1 <= k <= K. M <= P denotes retained
    future frequencies; M = P in the paper configuration.
    """

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02,
                 motion_size=(35, 66), device="cuda", padding=None,
                 EnableComplete=False, ddim_timesteps=100, scheduler='Linear',
                 model_type='data', mod_enable=True, mod_test=0.5,
                 dct=None, idct=None, n_pre=10):
        if int(noise_steps) != noise_steps or noise_steps < 1:
            raise ValueError("noise_steps must be a positive integer.")
        if int(ddim_timesteps) != ddim_timesteps or not 1 <= ddim_timesteps <= noise_steps:
            raise ValueError("ddim_timesteps must be an integer in [1, noise_steps].")
        if EnableComplete:
            raise ValueError("Future-only diffusion has no observation inpainting; set Complete=False.")
        self.noise_steps = int(noise_steps)
        self.beta_start = (1000 / self.noise_steps) * beta_start
        self.beta_end = (1000 / self.noise_steps) * beta_end
        self.motion_size = tuple(motion_size)
        self.device = torch.device(device)
        self.scheduler = scheduler
        transition_beta = self.prepare_noise_schedule().to(self.device)
        if not torch.isfinite(transition_beta).all() or not ((transition_beta > 0) & (transition_beta < 1)).all():
            raise ValueError("The schedule must have finite beta values strictly between 0 and 1.")
        self.beta = torch.cat((transition_beta.new_zeros(1), transition_beta))
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.ddim_timesteps = int(ddim_timesteps)

        # Legacy constructor fields remain readable, but padding and full-motion
        # DCT matrices never participate in the future-only diffusion process.
        self.model_type = model_type
        self.padding = padding
        self.EnableComplete = False
        self.mod_enable = mod_enable
        self.mod_test = mod_test
        self.dct = dct
        self.idct = idct
        self.n_pre = n_pre

        # Exactly S distinct steps ending at K, including S=1, S=K, K%S!=0.
        # The last reverse transition always reaches the clean index 0.
        steps = np.arange(1, self.ddim_timesteps + 1, dtype=np.int64)
        self.ddim_timestep_seq = (steps * self.noise_steps + self.ddim_timesteps - 1) // self.ddim_timesteps
        self.ddim_timestep_prev_seq = np.append(np.array([0]), self.ddim_timestep_seq[:-1])

    def prepare_noise_schedule(self):
        if self.scheduler == 'Linear':
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps).clamp(max=0.999)
        if self.scheduler == 'Cosine':
            return cosine_beta_schedule(self.noise_steps)
        if self.scheduler == 'Sqrt':
            return sqrt_beta_schedule(self.noise_steps)
        if self.scheduler == 'Sigmoid':
            return sigmoid_beta_schedule(self.noise_steps)
        if self.scheduler == 'Logarithmic':
            return logarithmic_beta_schedule(self.noise_steps, self.beta_start, self.beta_end)
        raise NotImplementedError(f"unknown scheduler: {self.scheduler}")

    def noise_motion(self, x, t):
        """Sample q(z_k | z_0); x [B,M,3J], t [B] in [0,K]."""
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError("t must contain one diffusion index per batch item.")
        if t.dtype != torch.long or torch.any(t < 0) or torch.any(t > self.noise_steps):
            raise ValueError("Diffusion indices must be torch.long values in [0, noise_steps].")
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        """Draw training indices uniformly from {1,...,K}, including K."""
        return torch.randint(low=1, high=self.noise_steps + 1, size=(n,))

    def sample_ddim_progressive(self, model, traj_dct, traj_dct_mod, mode_dict, noise=None):
        """Yield future-only latents [sample_num,M,3J] using deterministic DDIM.

        traj_dct must be None: this compatibility slot formerly held an
        observation-inpainting target. Conditions are [B,T,D_cond] or None;
        neither ground-truth future frames nor a temporal mask are consumed.
        """
        if traj_dct is not None:
            raise ValueError("traj_dct must be None: future-only sampling has no inpainting target.")
        sample_num = int(mode_dict['sample_num'])
        if sample_num < 1:
            raise ValueError("sample_num must be positive.")
        expected_shape = (sample_num, *self.motion_size)
        if noise is not None:
            if tuple(noise.shape) != expected_shape:
                raise ValueError(f"Expected future noise shape {expected_shape}, got {tuple(noise.shape)}.")
            x = noise.to(self.device)
        else:
            dtype = traj_dct_mod.dtype if traj_dct_mod is not None else torch.float32
            x = torch.randn(expected_shape, device=self.device, dtype=dtype)
        model.eval()
        with torch.no_grad():
            for i in reversed(range(self.ddim_timesteps)):
                t = torch.full((sample_num,), int(self.ddim_timestep_seq[i]), device=self.device, dtype=torch.long)
                prev_t = torch.full((sample_num,), int(self.ddim_timestep_prev_seq[i]), device=self.device, dtype=torch.long)
                alpha_hat = self.alpha_hat[t][:, None, None]
                alpha_hat_prev = self.alpha_hat[prev_t][:, None, None]
                predicted_noise = model(x, t, mod=traj_dct_mod)
                predicted_x0 = (x - torch.sqrt(1. - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat)
                pred_dir_xt = torch.sqrt(1. - alpha_hat_prev) * predicted_noise
                x = torch.sqrt(alpha_hat_prev) * predicted_x0 + pred_dir_xt
                yield x

    def sample_ddim(self, model, traj_dct, traj_dct_mod, mode_dict, noise=None):
        """Return future DCT coefficients [sample_num,M,3J].

        Pass traj_dct=None, observed DCT_T as traj_dct_mod, and a mode_dict
        with sample_num. Optional future-shaped noise makes sampling repeatable.
        """
        final = None
        for sample in self.sample_ddim_progressive(model, traj_dct, traj_dct_mod,
                                                   mode_dict, noise=noise):
            final = sample
        return final
