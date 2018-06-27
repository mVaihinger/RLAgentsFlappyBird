import multiprocessing
from a2c_smac_wrapper import a2c_smac_wrapper, arg_parser
import datetime, os

def main():
    args = arg_parser()
    print("rhs: " + str(args.run_parallel))
    optimized_config = a2c_smac_wrapper(**args.__dict__)

if __name__ == '__main__':
    main()
