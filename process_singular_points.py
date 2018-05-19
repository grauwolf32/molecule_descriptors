import os
import sys
import argparse
import subprocess

from subprocess import Popen, PIPE

def run_task(module_args, module_path, n_jobs=1):
    if not os.path.isfile(module_path):
        errlogger.error(("Module {} do not exist\n".format(module_path)))
        return ('','')

    module_dir  = os.path.dirname(module_path)
    args = ('parallel', '--gnu', '-a', module_args, '-j', str(n_jobs), '--colsep', ' ', module_path)
    output = Popen(args).communicate()

    return output

def main():
    parser = argparse.ArgumentParser(description='Admin Checker')
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--probe_radius', type=float, default=0.05)
    parser.add_argument('--space_sigma' , type=float, default=0.33)

    parser.add_argument('--sigma_min', type=float, default=0.36)
    parser.add_argument('--sigma_max', type=float, default=0.64)
    parser.add_argument('--n_sigmas' , type=int,   default=5)
    parser.add_argument('--n_jobs'   , type=int,   default=8)

    args = parser.parse_args()
    
    molecule_files = ["".join((args.path, f)) for f in os.listdir(args.path) if os.path.isfile("".join((args.path, f)))]
    molecule_files = filter(lambda x: x.split(".")[-1] == "wrl", molecule_files)

    with open("parallels_tmp", "w") as f:
        for mf in molecule_files:
            tmp = "--filename {} --probe_radius {} --space_sigma {} --sigma_min {} --sigma_max {} --n_sigmas {}".format(mf, args.probe_radius, args.space_sigma, args.sigma_min, args.sigma_max, args.n_sigmas) 
            f.write(tmp + "\n")

    run_task("./parallels_tmp", "./find_singular_points.py", n_jobs=args.n_jobs)

    #os.remove("./parallels_tmp")
    
if __name__ == "__main__":
    main()