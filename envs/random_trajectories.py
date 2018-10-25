import numpy as np
import os
import scipy.signal as spsig
import matplotlib.pyplot as plt
# import pickle
import simplejson
import time
PLOTTING = True

# TODO change the PATH!!
class Nonstationarity():
    def __init__(self, ns_type, seed, phase):
        self.points_list = []
        self.ns_type = ns_type
        self.phase = phase  # train or test
        if phase == 'test':
            self.seed = 100+seed
        else:
            self.seed = seed
        self.load_trajectory()

    def load_trajectory(self):
        path = os.getcwd()
        # path = '/media/mara/OS/Users/Mara/Documents/Masterthesis/RLAgents/RLAgentsFlappyBird'
        print('Load {}/envs/{}_envs/{}_{}.txt'.format(path, self.phase, self.ns_type, self.seed))
        with open(path + '/envs/{}_envs/{}_{}.txt'.format(self.phase, self.ns_type, self.seed), 'r') as f:
        # print('Load /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/envs/{}_envs/{}_{}.txt'.format(self.phase, self.ns_type, self.seed))
        # with open('/home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/envs/{}_envs/{}_{}.txt'.format(self.phase, self.ns_type, self.seed), 'r') as f:
            self.points_list = simplejson.load(f)

    def get_next_value(self):
        if len(self.points_list) <= 1:
            self.load_trajectory()
        return self.points_list.pop()


# class Filtered_RandomWalk():
#     def __init__(self, nsamples, offset, amplitude, fband):
#         self.nsamples = nsamples
#         self.points_list = []
#         self.filter_order = 4
#         self.fband = np.array(fband, dtype=float)
#         self.amplitude = amplitude
#         self.offset = offset
#
#     def add_values(self):
#         # Create random data which has mean 0 and std = amplitude
#         points = list(np.random.rand(self.nsamples) * self.amplitude - 0.5 * self.amplitude)  # Numbers between 0-1,
#         b, a = spsig.butter(self.filter_order, self.fband, btype='bandpass', analog=False, output='ba')
#         points = list(spsig.lfilter(b, a, points) + self.offset)
#         self.points_list = points
#         # plt.figure()
#         # plt.plot(self.points_list, 'r')
#         # plt.show()
#
#     def get_next_value(self):
#         if self.get_len() < 1:
#             self.add_values()
#         return self.points_list.pop()
#
#     def get_len(self):
#         return len(self.points_list)


class Overlayed_RandomSines():
    def __init__(self, nsamples, offset, amplitude, fband):
        self.nsamples = nsamples
        self.nsines = 30
        self.points_list = []
        self.fband = np.array(fband, dtype=float)
        self.amplitude = amplitude / self.nsines
        # print(self.amplitude)
        self.offset = offset

    def add_values(self):
        # Overlay 30 different sinus curves with each having a random phase and frequency:
        sample = np.linspace(0, self.nsamples)
        phases = np.random.rand(self.nsines)
        frequencies = self.fband[0] + np.random.rand(self.nsines) * (self.fband[1] - self.fband[0])
        amplitudes = (np.random.rand(self.nsines) - 0.5) * self.amplitude
        # print(amplitudes)

        # y = A * sin(2ft * pi - phase_rad)
        sin_func = lambda x, ampl, frq, ph: ampl * np.sin(2. * np.pi * (frq * x - ph))

        sine_waves = []
        plt.figure()
        for i in range(self.nsines):
            wave = [sin_func(x, amplitudes[i], frequencies[i], phases[i]) for x in
                    np.linspace(0, self.nsamples, self.nsamples)]
            plt.plot(wave)
            sine_waves.append(wave)

        self.points_list = list(np.sum(sine_waves, axis=0))
        self.points_list = [p + self.offset for p in self.points_list]
        if PLOTTING:
            plt.figure()
            plt.plot(self.points_list)
            plt.ylabel('gravity')
            plt.xlabel('# changes')
            plt.show()

    def get_next_value(self):
        if self.get_len() < 1:
            self.add_values()
        return self.points_list.pop()

    def get_len(self):
        return len(self.points_list)


class RandomIntSteps():
    def __init__(self, nsamples, time_interval, value_interval):
        self.nsamples = nsamples
        self.points_list = []
        self.time_interval = time_interval
        self.value_interval = value_interval

    def add_values(self):
        # create random switching time points (switch after 5 to 20 episodes)
        switching_time = list(np.arange(self.time_interval[0], self.time_interval[1], self.time_interval[2]))
        # repetitions = list(np.random.randint(low=self.time_interval[0], high=self.time_interval[1],
        #                                      size=int(self.nsamples / np.mean(self.time_interval))))
        repetitions = list(np.random.choice(switching_time, size=int(self.nsamples / np.mean(self.time_interval))))

        # create values of each step
        values = list(np.random.randint(low=self.value_interval[0], high=self.value_interval[1],
                                        size=int(self.nsamples / np.mean(self.time_interval))))

        points = []
        for i in range(len(repetitions)):
            points += [int(values[i]) for _ in range(repetitions[i])]
        self.points_list = points
        if PLOTTING:
            plt.figure()
            plt.plot(self.points_list, 'r')
            plt.ylabel('background speed')
            plt.xlabel('episode index')
            plt.show()

    def get_next_value(self):
        if self.get_len() < 1:
            self.add_values()
        return self.points_list.pop()

    def get_len(self):
        return len(self.points_list)

class RandomFloatSteps():
    def __init__(self, nsamples, time_interval, value_interval):
        self.nsamples = nsamples
        self.points_list = []
        self.time_interval = time_interval
        self.amplitude = abs(value_interval[0] - value_interval[1])
        # print(self.amplitude)
        self.offset = np.mean(value_interval)
        # print(self.offset)

    def add_values(self):
        # create random switching time points (switch after 5 to 20 episodes)
        switching_time = list(np.arange(self.time_interval[0],self.time_interval[1]+1,self.time_interval[2]))
        # print(switching_time)
        # repetitions = list(np.random.randint(low=self.time_interval[0], high=self.time_interval[1],
        #                                      size=int(self.nsamples / np.mean(self.time_interval))))
        repetitions = list(np.random.choice(switching_time, size=int(self.nsamples / np.mean(self.time_interval))))
        # print(repetitions)
        # create values of each step
        values = list(np.random.rand(int(self.nsamples / np.mean(self.time_interval))))
                                     # )low=self.value_interval[0], high=self.value_interval[1],
                                     #    size=int(self.nsamples / np.mean(self.time_interval))))
        # print(values)
        values = [round((self.amplitude * (v - 0.5) + self.offset), 1) for v in values]
        # print(values)
        points = []
        for i in range(len(repetitions)):
            points += [values[i] for _ in range(repetitions[i])]
        self.points_list = points

        if PLOTTING:
            plt.figure()
            plt.plot(self.points_list, 'r')
            plt.ylabel('background speed')
            plt.xlabel('episode index')
            plt.show()

    def get_next_value(self):
        if self.get_len() < 1:
            self.add_values()
        return self.points_list.pop()

    def get_len(self):
        return len(self.points_list)

def load_ns_trajectory(ns_name, seed):
    with open('train_envs/{}_{}.txt'.format(ns_name, seed), 'r') as f:
        ns = simplejson.load(f)
    return ns

def generate_trajectories(i):
    total_train_steps = 2e6 + 3
    total_test_steps = 5e5 + 3

    # Timing:
    #
    # In the episodic version for meta learning, each episode takes 60 steps ~ 4 tunnels ~ 30s
    # It is important that the discrete nonstationarities switch values at time steps, that are
    # a multiple of T_{episode} = 60 steps.
    # The mapping from #steps to time interval assumes dt=500ms

    #  --- gfNS ---
    # values:            [0.1, 0.2, ..., 1.4, 1.5]
    # time_interval:     [120 steps, 180steps, 240 steps]
    # which is equivalent to 1, 1.5 and 2 mins (dt = 500ms) or about 8, 12, 16 tunnels

    # --- gsNS ---
    # values:            [0.1, ..., 1.5]
    # freqs of sines:    [0.000012, ..., 0.000023]
    # equiv. period T:   [43200 steps, ..., 86400 steps] ~ [6h - 12h]

    # --- bfNS ---
    # values:            [2,3,4,5,6]
    # time_interval:     [120 steps, 180steps, 240 steps]
    # which is equivalent to 1, 1.5 and 2 mins (dt = 500ms) or about 8, 12, 16 tunnels

    # --- random state feature ---
    # values:            [0.1, ..., 1.]
    # freqs of sines:    [0.00167, ..., 0.05]
    # equiv. period T:   [20 steps, ..., 600 steps] ~ [10s - 5mins]

    np.random.seed(i)
    # # gfNS
    # print('gfNS')
    # gfNS = RandomFloatSteps(nsamples=total_train_steps, time_interval=[120, 240, 60],
    #                         value_interval=[0.5, 1.5])  # upper bound is excluded
    # gfNS.add_values()
    # with open('train_envs/gfNS_{}.txt'.format(i), 'w') as f:
    #     simplejson.dump(gfNS.points_list, f)
    #
    # # gsNS
    # print('gsNS')
    # gsNS = Overlayed_RandomSines(nsamples=total_train_steps, offset=1., amplitude=5, fband=[0.000012, 0.000023])
    # gsNS.add_values()
    # with open('train_envs/gsNS_{}.txt'.format(i), 'w') as f:
    #     simplejson.dump(gsNS.points_list, f)
    #
    # # # bfNS
    # # print('bfNS')
    # # bfNS = RandomIntSteps(nsamples=total_train_steps, time_interval=[120, 240, 60],
    # #                       value_interval=[2, 7])  # upper bound is excluded
    # # bfNS.add_values()
    # # with open('train_envs/bfNS_{}.txt'.format(i), 'w') as f:
    # #     simplejson.dump(bfNS.points_list, f)
    #
    # random feature
    print('random feature')
    rand_feat = Overlayed_RandomSines(nsamples=total_train_steps, offset=0.5, amplitude=3., fband=[0.0017, 0.05])
    rand_feat.add_values()
    with open('train_envs/rand_feat_{}.txt'.format(i), 'w') as f:
        simplejson.dump(rand_feat.points_list, f)


    time.sleep(5)


    seed = i + 100
    np.random.seed(seed)
    # gfNS, 60
    # gfNS = RandomFloatSteps(nsamples=total_test_steps, time_interval=[120, 240, 60],  # 1-2 mins with dt = 500ms
    #                         value_interval=[0.5, 1.5])  # upper bound is excluded
    # gfNS.add_values()
    # with open('test_envs/gfNS_{}.txt'.format(seed), 'w') as f:
    #     simplejson.dump(gfNS.points_list, f)
    #
    # # gsNS
    # gsNS = Overlayed_RandomSines(nsamples=total_test_steps, offset=1., amplitude=5, fband=[0.000012, 0.000023])
    # gsNS.add_values()
    # with open('test_envs/gsNS_{}.txt'.format(seed), 'w') as f:
    #     simplejson.dump(gsNS.points_list, f)

    # # bfNS
    # bfNS = RandomIntSteps(nsamples=total_test_steps, time_interval=[120, 240, 60],  # 1-2 mins with dt = 500ms
    #                       value_interval=[2, 7])  # [5,20] [3,6]upper bound is excluded
    # bfNS.add_values()
    # with open('test_envs/bfNS_{}.txt'.format(seed), 'w') as f:
    #     simplejson.dump(bfNS.points_list, f)

    # random feature
    print('test random feature')
    rand_feat = Overlayed_RandomSines(nsamples=total_test_steps, offset=0.5, amplitude=3., fband=[0.0017, 0.05])
    rand_feat.add_values()
    with open('test_envs/rand_feat_{}.txt'.format(seed), 'w') as f:
        simplejson.dump(rand_feat.points_list, f)


if __name__ == '__main__':
    # param_traj = Nonstationarity('rand_feat', 1, 'train')
    # plt.figure()
    # plt.plot(param_traj.points_list)
    #
    # plt.figure()
    # param_traj = Nonstationarity('gsNS', 1, 'test')
    # plt.plot(param_traj.points_list)
    # plt.show()
    # print(param_traj.get_next_value())

    gfNS = RandomFloatSteps(nsamples=3000, time_interval=[120, 240, 60], value_interval=[0.5, 1.5]) # 1-2 mins with dt = 500ms
    #                           # upper bound is excluded
    gfNS.add_values()
    # for i in range(2):
    #     generate_trajectories(i+18)
    #     print('%s: sleep...'% (i+18))
    #     time.sleep(3)

    # ns = load_ns_trajectory('gfNS', 2)
    # test = Overlayed_RandomSines(nsamples=360000, offset=1., amplitude=5,
    #                              fband=[0.000012, 0.000023]) #]0.006, 0.01])  # 1/100 - 1/150 = 0.01 - 0.006
    # # 0.000023 --> ~ 43200 steps --> 360 mins --> 6h
    # # 0.000012 --> ~ 86400 steps --> 720 mins --> 12h
    # test.add_values()

    # test2 = random_speed = RandomIntSteps(nsamples=3000, time_interval=[120, 240],
    #                                       value_interval=[3, 6])  # upper bound is excluded  # TODO set ttrace_length
    # test2.add_values()

    # test2 = random_speed = RandomFloatSteps(nsamples=3600, time_interval=[120, 240],
    #                                         value_interval=[0.5,1.5])  # upper bound is excluded  # TODO set ttrace_length
    # test2.add_values()
