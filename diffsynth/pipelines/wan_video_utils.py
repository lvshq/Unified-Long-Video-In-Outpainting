"""Temporal-window helpers for the project's Wan video editor."""


def get_views(video_length, window_size, stride):
    """Cover a nonempty latent sequence, including clips shorter than one window."""
    if video_length <= 0 or window_size <= 0 or stride <= 0:
        raise ValueError("Video length, window size, and stride must be positive.")
    if stride > window_size:
        raise ValueError("Stride cannot exceed window size (it would leave uncovered frames).")
    if video_length <= window_size:
        return [(0, video_length)]
    starts = list(range(0, video_length - window_size + 1, stride))
    if starts[-1] + window_size < video_length:
        starts.append(video_length - window_size)
    return [(start, start + window_size) for start in starts]


def denoise_window(scheduler, predict, latents, timestep, step_index, high_order=True):
    """Apply the original explicit-midpoint solver, or Euler, including the final step.

    This is the midpoint implementation in the research code. It is a second-order
    Runge–Kutta method, distinct from the endpoint-averaged Heun method in the paper.
    ``predict`` accepts a latent tensor and a scalar diffusion timestep.
    """
    prediction = predict(latents, timestep)
    if not high_order:
        return scheduler.step(prediction, timestep, latents)
    midpoint = scheduler.step_mid(prediction, timestep, latents)
    next_timestep = (
        scheduler.timesteps[step_index + 1] if step_index + 1 < len(scheduler.timesteps) else 0.0
    )
    midpoint_prediction = predict(midpoint, (timestep + next_timestep) / 2)
    return scheduler.step_solver(midpoint_prediction, prediction, timestep, latents)


def blend_windows(latents, views, update):
    """Denoise each view and normalize Hamming weights along the temporal axis."""
    import torch

    values = torch.zeros_like(latents, dtype=torch.float32)
    weights = torch.zeros((1, 1, latents.shape[2], 1, 1), device=latents.device)
    for start, end in views:
        window = latents[:, :, start:end].float()
        blend = torch.hamming_window(
            end - start, periodic=False, device=latents.device, dtype=torch.float32
        ).view(1, 1, -1, 1, 1)
        values[:, :, start:end] += update(window) * blend
        weights[:, :, start:end] += blend
    if torch.any(weights == 0):
        raise ValueError("Temporal windows must cover every frame.")
    return values / weights
