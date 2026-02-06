import safety_gymnasium

def register_safety_inverted_pendulum():
    safety_gymnasium.register(id="SafetyInvertedPendulum-v4",
        entry_point="../misc/safety_inverted_pendulum_v4:SafetyInvertedPendulumEnv",
        max_episode_steps=1000,
        reward_threshold=950.0,
    )