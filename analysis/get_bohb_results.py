import matplotlib.pyplot as plt
# import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis
import pickle, io, os

results_dir='/media/mara/OS/Users/Mara/Documents/Masterthesis/Results/BOHB_configs/lower_min_budget_08_14'

for mthd in ['gru_ppo', 'lstm_ppo']:  # 'ppo', 'a2c', 'lstm_a2c', 'gru_a2c', 'lstm_dqn', 'dqn']: # 'gru_dqn'
# for mthd in ['dqn']:
    with open(os.path.join(results_dir, (mthd + '_results.pkl')), 'rb') as f:
        result = pickle.load(f)

    print(mthd)
    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()

    # Here is how you get he incumbent (best configuration)
    inc_id = result.get_incumbent_id()

    # let's grab the run on the highest budget
    inc_runs = result.get_runs_by_id(inc_id)
    inc_run = inc_runs[-1]

    # We have access to all information: the config, the loss observed during
    # optimization, and all the additional information
    inc_losses = [run.loss for run in inc_runs]
    inc_loss = inc_run.loss
    inc_config = id2conf[inc_id]['config']
    # inc_test_loss = inc_run.info['test accuracy']

    print('Best found configuration:')
    print(inc_config)
    print(1-inc_loss)
    print(inc_losses)

    # Let's plot the observed losses grouped by budget,
    # get all executed runs
    all_runs = result.get_all_runs()
    hpvis.losses_over_time(all_runs)
    plt.savefig(os.path.join(results_dir, (mthd+'_losses_over_time.pdf')))

    # # the number of concurent runs,
    # hpvis.concurrent_runs_over_time(all_runs)
    #
    # # and the number of finished runs.
    # hpvis.finished_runs_over_time(all_runs)
    #
    # # This one visualizes the spearman rank correlation coefficients of the losses
    # # between different budgets.
    # hpvis.correlation_across_budgets(result)
    #
    # # For model based optimizers, one might wonder how much the model actually helped.
    # # The next plot compares the performance of configs picked by the model vs. random ones
    # hpvis.performance_histogram_model_vs_random(all_runs, id2conf)

    # plt.show()

    print(result)