from gym.envs.registration import register
import gym.spaces

for game in ['FlappyBird', 'ContFlappyBird']:
    nondeterministic = False
    register(
        id='{}-v1'.format(game),
        entry_point='envs.environment:PLEEnv_state',
        kwargs={'game_name': game, 'display_screen': False},
        tags={'wrapper_config.TimeLimit.max_episode_steps': 10000000},
        nondeterministic=nondeterministic,)

    register(
        id='{}-v2'.format(game),
        entry_point='envs.environment:PLEEnv_state',
        kwargs={'game_name': game, 'display_screen': False, 'nonstationary': True},
        tags={'wrapper_config.TimeLimit.max_episode_steps': 10000000},
        nondeterministic=nondeterministic, )

    register(
        id='{}-v3'.format(game),
        entry_point='envs.environment:PLEEnv_state',
        kwargs={'game_name': game, 'display_screen': False},
        tags={'wrapper_config.TimeLimit.max_episode_steps': 7000},
        nondeterministic=nondeterministic, )

    register(
        id='{}-v4'.format(game),
        entry_point='envs.environment:PLEEnv_state',
        kwargs={'game_name': game, 'display_screen': False, 'nonstationary': True},
        tags={'wrapper_config.TimeLimit.max_episode_steps': 7000},
        nondeterministic=nondeterministic, )

# TODO add environments which only have fast or slow changing parameters.

# distances of pipes are 0.5*width with width = 288. The first pipe has a distance of 207 px.
# So 200 pipes would have a distance of 207+200*144 = 29007 px to the inital position of the agent.
# In every timestep the agent does 4 px steps (speed*scale) - in the non-stationary environment this value changes.
# In the stationay case the episode should stop after 29007/4 px-steps which are 7251 timesteps.
# So 2000 entspricht 50 pipes.
# But maybe it's a better idea to stop the environment after some time rather than after some amoutn of pipes.

