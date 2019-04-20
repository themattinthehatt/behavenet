import os
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', '-r', help='results directory')
    args = parser.parse_args()

    hyper_search(args.data_dir,args.dest_dir)
