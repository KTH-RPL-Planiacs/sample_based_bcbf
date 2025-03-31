from .simulation_env import ColAvEnv, TrackingEnv

def get_env(env_name: str, config: dict):
    if "colav_env" in env_name:
        return ColAvEnv(config)
    elif "tracking_env" in env_name:
        return TrackingEnv(config)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
