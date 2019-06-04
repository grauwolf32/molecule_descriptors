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
    parser.add_argument('--n_jobs'   , type=int,   default=8)
    parser.add_argument('--n_start', type=int, required=True)
    parser.add_argument('--n_end', type=int, required=True)
    args = parser.parse_args()
    
    molecule_files = ["".join((args.path, f)) for f in os.listdir(args.path) if os.path.isfile("".join((args.path, f)))]
    molecule_files = filter(lambda x: x.split(".")[-1] == "wrl", molecule_files)
    molecule_files = sorted(molecule_files)[args.n_start:args.n_end]

    with open("parallels_tmp", "w") as f:
        for mf in molecule_files:
            tmp = "-f {}".format(mf) 
            f.write(tmp + "\n")

    run_task("./parallels_tmp", "./find_sp.py", n_jobs=args.n_jobs)

    #os.remove("./parallels_tmp")
    
if __name__ == "__main__":
    main()