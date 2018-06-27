import multiprocessing
from dqn_smac_wrapper import dqn_smac_wrapper, arg_parser
import datetime, os

def main():
    args = arg_parser()
    print("rhs: " + str(args.run_parallel))
    optimized_config = dqn_smac_wrapper(**args.__dict__)

if __name__ == '__main__':
    main()