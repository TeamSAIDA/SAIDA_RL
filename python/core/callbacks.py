# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

from core.common.callback import *
import datetime
from core.common.util import *

class TestLogger(Callback):
    """ Logger Class for Test """
    def on_train_begin(self, logs={}):
        """ Print logs at beginning of training"""
        print('Testing for {} episodes ...'.format(self.params['nb_episodes']))

    def on_episode_end(self, episode, logs={}):
        """ Print logs at end of each episode """
        template = 'Episode {0}: reward: {1:.3f}, steps: {2}'
        variables = [
            episode + 1,
            logs['episode_reward'],
            logs['nb_steps'],
        ]
        print(template.format(*variables))


class TrainEpisodeLogger(Callback):
    def __init__(self):
        self.episode_start = {}
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.step = 0
        self.train_start = 0

    def on_train_begin(self, logs={}):
        """ Print training values at beginning of training """
        self.train_start = timeit.default_timer()

    def on_train_end(self, logs={}):
        """ Print training time at end of training """
        duration = timeit.default_timer() - self.train_start
        print('done, took {:.3f} seconds'.format(duration))

    def on_episode_begin(self, episode, logs={}):
        """ Reset environment variables at beginning of each episode """
        self.episode_start[episode] = timeit.default_timer()
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []

    def on_episode_end(self, episode, logs={}):
        """ Compute and print training statistics of the episode when done """
        duration = timeit.default_timer() - self.episode_start[episode]
        episode_steps = len(self.observations[episode])

        nb_step_digits = str(int(np.ceil(np.log10(self.params['nb_steps']))) + 1)
        template = '{step: ' + nb_step_digits + 'd}/{nb_steps}: episode: {episode}, duration: {duration:.3f}s, episode steps: {episode_steps}, steps per second: {sps:.0f}, episode reward: {episode_reward:.3f}, mean reward: {reward_mean:.3f} [{reward_min:.3f}, {reward_max:.3f}], mean action: {action_mean:.3f} [{action_min:.3f}, {action_max:.3f}]' #, mean observation: {obs_mean:.3f} [{obs_min:.3f}, {obs_max:.3f}]'
        variables = {
            'step': self.step,
            'nb_steps': self.params['nb_steps'],
            'episode': episode + 1,
            'duration': duration,
            'episode_steps': episode_steps,
            'sps': float(episode_steps) / duration,
            'episode_reward': np.sum(self.rewards[episode]),
            'reward_mean': np.mean(self.rewards[episode]),
            'reward_min': np.min(self.rewards[episode]),
            'reward_max': np.max(self.rewards[episode]),
            'action_mean': np.mean(self.actions[episode]),
            'action_min': np.min(self.actions[episode]),
            'action_max': np.max(self.actions[episode]),
            # 'obs_mean': np.mean(self.observations[episode]),
            # 'obs_min': np.min(self.observations[episode]),
            # 'obs_max': np.max(self.observations[episode])
        }
        print(template.format(**variables))

        # Free up resources.
        del self.episode_start[episode]
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]

    def on_step_end(self, step, logs={}):
        """ Update statistics of episode after each step """
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        self.step += 1


class TrainIntervalLogger(Callback):
    def __init__(self, interval=10000):
        self.interval = interval
        self.step = 0
        self.reset()
        # self.train_start = 0


    def reset(self):
        """ Reset statistics """
        self.interval_start = timeit.default_timer()
        self.progbar = Progbar(target=self.interval)
        self.infos = []
        self.info_names = None
        self.episode_rewards = []

    def on_train_begin(self, logs={}):
        """ Initialize training statistics at beginning of training """
        self.train_start = timeit.default_timer()
        print('Training for {} steps ...'.format(self.params['nb_steps']))

    def on_train_end(self, logs={}):
        """ Print training duration at end of training """
        duration = timeit.default_timer() - self.train_start
        print('done, took {:.3f} seconds'.format(duration))

    def on_step_begin(self, step, logs={}):
        """ Print metrics if interval is over """
        if self.step % self.interval == 0:
            if len(self.episode_rewards) > 0:
                formatted_metrics = ''
                formatted_infos = ''
                if len(self.infos) > 0:
                    infos = np.array(self.infos)
                    # if not np.isnan(infos).all():  # not all values are means
                    # if not pd.isnull(infos).all():  # not all values are means
                    #     means = np.nanmean(self.infos, axis=0)
                    #     assert means.shape == (len(self.info_names),)
                    #     for name, mean in zip(self.info_names, means):
                    #         formatted_infos += ' - {}: {:.3f}'.format(name, mean)
                print('{} episodes - episode_reward: {:.3f} [{:.3f}, {:.3f}]'.format(len(self.episode_rewards), np.mean(self.episode_rewards), np.min(self.episode_rewards), np.max(self.episode_rewards)))
                print('')
            self.reset()
            print('Interval {} ({} steps performed)'.format(self.step // self.interval + 1, self.step))

    def on_step_end(self, step, logs={}):
        """ Update progression bar at the end of each step """
        if self.info_names is None:
            self.info_names = logs['info'].keys()
        values = [('reward', logs['reward'])]
        if KERAS_VERSION > '2.1.3':
            self.progbar.update((self.step % self.interval) + 1, values=values)
        else:
            self.progbar.update((self.step % self.interval) + 1, values=values, force=True)
        self.step += 1
        if len(self.info_names) > 0:
            self.infos.append([logs['info'][k] for k in self.info_names])

    def on_episode_end(self, episode, logs={}):
        """ Update reward value at the end of each episode """
        self.episode_rewards.append(logs['episode_reward'])


class FileLogger(Callback):
    def __init__(self, filepath, interval=None):
        self.filepath = filepath
        self.interval = interval

        # Some algorithms compute multiple episodes at once since they are multi-threaded.
        # We therefore use a dict that maps from episode to metrics array.
        self.starts = {}
        self.data = {}

    def on_train_begin(self, logs={}):
        """ Initialize model metrics before training """
        pass

    def on_train_end(self, logs={}):
        """ Save model at the end of training """
        self.save_data()

    def on_episode_begin(self, episode, logs={}):
        """ Initialize metrics at the beginning of each episode """
        assert episode not in self.metrics
        assert episode not in self.starts
        self.starts[episode] = timeit.default_timer()

    def on_episode_end(self, episode, logs={}):
        """ Compute and print metrics at the end of each episode """
        duration = timeit.default_timer() - self.starts[episode]

        data = list(logs.items())
        data += [('episode', episode), ('duration', duration)]
        for key, value in data:
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

        if self.interval is not None and episode % self.interval == 0:
            self.save_data()

        # Clean up.
        del self.starts[episode]

    def on_step_end(self, step, logs={}):
        """ Append metric at the end of each step """
        self.metrics[logs['episode']].append(logs['metrics'])

    def save_data(self):
        """ Save metrics in a json file """
        if len(self.data.keys()) == 0:
            return

        # Sort everything by episode.
        assert 'episode' in self.data
        sorted_indexes = np.argsort(self.data['episode'])
        sorted_data = {}
        for key, values in self.data.items():
            assert len(self.data[key]) == len(sorted_indexes)
            # We convert to np.array() and then to list to convert from np datatypes to native datatypes.
            # This is necessary because json.dump cannot handle np.float32, for example.
            sorted_data[key] = np.array([self.data[key][idx] for idx in sorted_indexes]).tolist()

        # Overwrite already open file. We can simply seek to the beginning since the file will
        # grow strictly monotonously.
        with open(self.filepath, 'w') as f:
            json.dump(sorted_data, f)


class ModelIntervalCheckpoint(Callback):
    def __init__(self, filepath, step_interval=None, episode_interval=None, condition=None, condition_count=0, verbose=0
                 , **kwargs):
        super(ModelIntervalCheckpoint, self).__init__(**kwargs)
        self.filepath = filepath
        self.step_interval = step_interval
        self.episode_interval = episode_interval
        self.condition = condition
        self.condition_count = condition_count
        self.continuous_condition_count = 0
        self.verbose = verbose
        self.total_steps = 0

    def on_step_end(self, step, logs={}):
        if self.agent is None:
            raise Exception('agent is necessary.')

        if self.step_interval is None:
            return
        """ Save weights at interval steps during training """
        self.total_steps += 1
        if self.total_steps % self.step_interval != 0:
            # Nothing to do.
            return

        filepath = self.filepath.format(step=self.total_steps, **logs)
        if self.verbose > 0:
            print('Step {}: saving model to {}'.format(self.total_steps, filepath))
        self.agent.save_weights(filepath, overwrite=True)

    def on_episode_end(self, episode, logs={}):
        if self.agent is None:
            raise Exception('agent is necessary.')

        if callable(self.condition):
            if self.condition(episode, logs):
                self.continuous_condition_count += 1
            else:
                self.continuous_condition_count = 0

            if self.continuous_condition_count > self.condition_count:
                filepath = self.filepath.format(episode=episode, **logs)
                if self.verbose > 0:
                    print('continuous condition episode {}: saving model to {}'.format(episode, filepath))
                self.agent.save_weights(filepath, True)
                return

        if self.episode_interval is None or episode % self.episode_interval != 0 or episode == 0:
            return

        filepath = self.filepath.format(episode=episode, step='', **logs)

        if self.verbose > 0:
            print('episode {}: saving model to {}'.format(episode, filepath))

        self.agent.save_weights(filepath, True)


class History(Callback):
    """Callback that records events into a `History` object.

    This callback is automatically applied to every Keras model. The `History` object
    gets returned by the `fit` method of models.
    """

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)



class DrawTrainPlotCallback(Callback):

    def __init__(self, file_path=None, plot_interval=10000, data_for_plot=['episode_reward', 'nb_episode_steps']):
        """
        :param file_path: plot file path
        :param plot_interval: episode interval
        :param data_for_plot:  array for data to be plotted.
        """
        super(DrawTrainPlotCallback, self).__init__()
        self._file_path = file_path
        self._plot_interval = plot_interval
        self._data_for_plot = data_for_plot
        self._map_for_plot = {}

        # here we map data with its list to save data every episodes in iterations
        for d in data_for_plot:
            self._map_for_plot[d] = []

        if self._file_path == None:
            self._file_path = 'plot_{}_{}_{:%Y%m%d%H%M%S}.png'
        else:
            self._file_path = self._file_path + 'plot_{}_{}_{:%Y%m%d%H%M%S}.png'

    def on_episode_end(self, episode, logs):
        # cumulative data into list for plot in the end
        for d in self._data_for_plot:
            self._map_for_plot[d].append(logs[d])

        if (episode != 0) and (episode % self._plot_interval == 0):
            fig = plt.figure()

            for k in self._data_for_plot:
                plt.plot(self._map_for_plot[k], label=k)
                plt.title('episodic result')

            # outdent to create one plot file, it doesn't have to create two or more files. because drawing plot is too slow.
            feats = '_'.join(self._data_for_plot)
            fig.savefig(self._file_path.format(feats, episode, datetime.now()))



class DrawTrainMovingAvgPlotCallback(Callback):

    def __init__(self, file_path, plot_interval=10000, time_window=1000, l_label=['reward', 'kill_cnts', 'hps'], save_raw_data=False, title=''):
        """
        :param file_path: plot file path
        :param plot_interval: episode interval
        :param time_window " time window for moving average
        :param data_for_plot:  array for data to be plotted.
        """
        super(DrawTrainMovingAvgPlotCallback, self).__init__()
        self._file_path = file_path
        self._plot_interval = plot_interval
        self.l_label = l_label
        self._map_for_plot = {}
        self.time_window = time_window
        self.save_raw_data = save_raw_data
        self.title = title

        # in case when length of title is over 130, it can't display all so that split title into two lines.
        if len(self.title) > 120:
            self.title = self.title[:120] + '\n' + self.title[120:]

        # here we map data with its list to save data every episodes in iterations
        for l in l_label:
            self._map_for_plot[l] = []

        # if self._file_path == None:
        #     self._file_path = 'plot_{}_{}_{:%Y%m%d%H%M%S}.png'
        # else:
        #     self._file_path = self._file_path + 'plot_{}_{}_{:%Y%m%d%H%M%S}.png'

    def on_episode_end(self, episode, logs):
        """
        :param episode: episode index
        :param logs: logs is map containing value with its key which is also used as label.
        :return:
        """

        # cumulative data into list for plot in the end
        for l in self.l_label:
            if logs.get(l) is None:
                self._map_for_plot[l].append(logs['info'][l])
            else:
                self._map_for_plot[l].append(logs[l])

        if (episode != 0) and (episode % self._plot_interval == 0):

            for l in self.l_label:
                reward_moving_avg_plot(title=self.title, y_data_list=self._map_for_plot[l], window=self.time_window, label=l,
                                   filepath=self._file_path.format(l))

            if self.save_raw_data:
                with open(self._file_path.format('data') + ".pickle", 'wb') as save_fig_data:
                    pickle.dump([(l, self._map_for_plot[l]) for l in self.l_label], save_fig_data)

