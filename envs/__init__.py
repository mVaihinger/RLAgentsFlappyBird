from gym.envs.registration import register
import gym.spaces
import numpy as np

# for game in ['FlappyBird', 'ContFlappyBird']:
for game in ['ContFlappyBird']:
    nondeterministic = False
    register(
        id='{}-v1'.format(game),
        entry_point='envs.environment:PLEEnv_state',
        kwargs={'game_name': game, 'display_screen': False},
        tags={'wrapper_config.TimeLimit.max_episode_steps': 10000000},
        nondeterministic=nondeterministic,)

    # CLIPPED ENVIRONMENT
    register(
        id='{}-v3'.format(game),
        entry_point='envs.environment:PLEEnv_state',
        kwargs={'game_name': game, 'display_screen': False},
        tags={'wrapper_config.TimeLimit.max_episode_steps': 3000},  # TODO 600
        nondeterministic=nondeterministic, )

    for p in ['train', 'test']:
        # Random features
        for nrf in [0, 1, 2, 3, 4]:
            register(
                id='{}-nrf{}-{}-v0'.format(game, nrf, p),
                entry_point='envs.nenvironment:PLEEnv_state',
                kwargs={'game_name': game, 'display_screen': False, 'nrandfeatures': nrf, 'phase': p},
                tags={'wrapper_config.TimeLimit.max_episode_steps': 10000000},
                nondeterministic=nondeterministic, )

            register(
                id='{}-clip-nrf{}-{}-v0'.format(game, nrf, p),
                entry_point='envs.nenvironment:PLEEnv_state',
                kwargs={'game_name': game, 'display_screen': False, 'nrandfeatures': nrf, 'phase': p},
                tags={'wrapper_config.TimeLimit.max_episode_steps': 600},
                nondeterministic=nondeterministic, )

            # Noise
            for nl in [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3,
                       0.35, 0.4, 0.5]:
                register(
                    id='{}-nl{}-nrf{}-{}-v0'.format(game, nl, nrf, p),
                    entry_point='envs.nenvironment:PLEEnv_state',
                    kwargs={'game_name': game, 'display_screen': False, 'noise_level': nl},
                    tags={'wrapper_config.TimeLimit.max_episode_steps': 10000000},
                    nondeterministic=nondeterministic, )

                register(
                    id='{}-clip-nl{}-nrf{}-{}-v0'.format(game, nl, nrf, p),
                    entry_point='envs.nenvironment:PLEEnv_state',
                    kwargs={'game_name': game, 'display_screen': False, 'noise_level': nl},
                    tags={'wrapper_config.TimeLimit.max_episode_steps': 600},
                    nondeterministic=nondeterministic, )

            # nonstationary
            for ns in ['gfNS', 'gsNS', 'bfNS', 'hNS']:
                register(
                    id='{}-{}-nrf{}-{}-v0'.format(game, ns, nrf, p),
                    entry_point='envs.nenvironment:PLEEnv_state',
                    kwargs={'game_name': game, 'display_screen': False, 'nonstationary': ns, 'nrandfeatures': nrf, 'phase': p},
                    tags={'wrapper_config.TimeLimit.max_episode_steps': 10000000},
                    nondeterministic=nondeterministic, )

                register(
                    id='{}-clip-{}-nrf{}-{}-v0'.format(game, ns, nrf, p),
                    entry_point='envs.nenvironment:PLEEnv_state',
                    kwargs={'game_name': game, 'display_screen': False, 'nonstationary': ns, 'nrandfeatures': nrf, 'phase': p},
                    tags={'wrapper_config.TimeLimit.max_episode_steps': 600},
                    nondeterministic=nondeterministic, )

        # # nonstationary
        # for ns in ['gfNS', 'gsNS', 'hNS']:
        #     register(
        #         id='{}-{}-{}-v0'.format(game, ns, p),
        #         entry_point='envs.nenvironment:PLEEnv_state',
        #         kwargs={'game_name': game, 'display_screen': False, 'nonstationary': ns, 'phase': p},
        #         tags={'wrapper_config.TimeLimit.max_episode_steps': 10000000},
        #         nondeterministic=nondeterministic, )
        #
        #     register(
        #         id='{}-clip-{}-{}-v0'.format(game, ns, p),
        #         entry_point='envs.nenvironment:PLEEnv_state',
        #         kwargs={'game_name': game, 'display_screen': False, 'nonstationary': ns, 'phase': p},
        #         tags={'wrapper_config.TimeLimit.max_episode_steps': 600},
        #         nondeterministic=nondeterministic, )

# distances of pipes are 0.5*width with width = 288. The first pipe has a distance of 207 px.
# So 200 pipes would have a distance of 207+200*144 = 29007 px to the inital position of the agent.
# In every timestep the agent does 4 px steps (speed*scale) - in the non-stationary environment this value changes.
# In the stationay case the episode should stop after 29007/4 px-steps which are 7251 timesteps.
# So 2000 entspricht 50 pipes.
# But maybe it's a better idea to stop the environment after some time rather than after some amoutn of pipes.

