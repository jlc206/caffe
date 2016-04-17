import os
import sys
import argparse
import glob
from datetime import datetime, timedelta

if __name__ == "__main__":

    #PARSE ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument('SOLVER_PROTOTXT')
    parser.add_argument('GPUS')
    args = parser.parse_args()
    
    #CREATE DIR TO STORE LOGS
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    #RUN TRAIN
    for i in range(1,4):
        opt = args.GPUS
        command = "caffe train --solver=" + args.SOLVER_PROTOTXT + " -gpu " + opt + " 2>&1 | tee -a logs/gpu" + opt + "_run" + str(i) + ".log"
        os.system(command)  
        
    print
    print "Iteration times for " + args.SOLVER_PROTOTXT + " with GPU(S) " + opt + "..."
    #PARSE THE LOGS AND GET AVG ITERATION TIME
    filename = "logs/gpu" + opt + ".log"
    
    for filename in glob.glob('logs/gpu' + opt + '*.log'):
        print "parsing " + filename
        diplay = 1 #THIS SHOULD BE HANDLED DIFFERENTLY IN THE FUTURE TO WORK FOR ANY PROTOTXT?
        log = open(filename, "r")
        solving = False
        times = []
        comm = []

        for line in log.readlines():
            tokens = line.rstrip().split()
            if tokens[0] == 'display:':  #Display every n iterations
                display = int(tokens[1])
            if len(tokens) > 5 and tokens[4] == 'Solving':
                solving = True
            if solving:
                if len(tokens) > 10 and tokens[4] == 'Iteration' and tokens[6] == 'loss':
                    times.append(tokens[1])
                    comm.append(float(tokens[12]))
        comm.pop(0)

        tdeltas = []
        fmt = '%H:%M:%S.%f'
        for i in range(len(times)-1):
            tdeltas.append(datetime.strptime(times[i+1], fmt) - datetime.strptime(times[i], fmt))

        avg_tdelta = (sum(tdeltas, timedelta()) / (len(tdeltas)*display)).total_seconds() * 1000
        avg_communication = sum(comm)/(len(comm)*display)
        avg_computation = avg_tdelta - avg_communication
        print "Average time for forward/backward pass (per iteration): " + str(avg_computation) + " ms."
        print "Average time for GPU communication (per iteration): " + str(avg_communication) + " ms." #already in ms
