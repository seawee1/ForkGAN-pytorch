import argparse
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing

"""
This script creates a loss plot from a ForkGAN loss_log.txt.
"""

# Example call: python loss_log_to_plot.py path/to/loss_log.txt --smoothing 0.1
# If you want to plot only specific losses: python loss_log_to_plot.py path/to/loss_log.txt --smoothing 0.1 --looses G_A_inst,G_A,D_A,D_A_inst
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(dest='log_file', type=str, help='loss_log.txt path')
parser.add_argument('--losses', default='', type=str, help='Names of losses to plot')
parser.add_argument('--smoothing', default=0.1, type=float, help='Exponential weighting coefficiant for graph smoothing')
args = parser.parse_args()

log_file = args.log_file
print(f'Plotting {log_file}...')
# Find out loss names to plot
loss_names = ['D_A', 'G_A', 'G_A_rec', 'G_A_fake_rec', 'cycle_A', 'idt_A', 'A_rec', \
                   'D_B', 'G_B', 'G_B_rec', 'G_B_fake_rec', 'cycle_B', 'idt_B', 'B_rec', \
                   'G_DC', 'Perc', 'DC']
if args.losses != '':
    loss_names = args.losses.split(',')
loss_dict = {x: [] for x in loss_names}

# Open loss_log.txt
f = open(log_file, 'r')
iter_1, iter_2, iter_step = None, None, None
for line in f.readlines():
    if 'D_A' in line: # D_A loss should be in every loss print line. Hacky but works
        tokens = line.split(' ')

        if iter_1 is None:
            iter_1 = float(tokens[tokens.index('iters:')+1][:-1])
        elif iter_2 is None:
            iter_2 = float(tokens[tokens.index('iters:')+1][:-1])
            iter_step = iter_2 - iter_1

        for loss in loss_names:
            i = tokens.index(f'{loss}:')
            loss_dict[loss].append(float(tokens[i+1]))

# Exponentially smooth every graph and plot
for loss in loss_names:
    exp = ExponentialSmoothing(loss_dict[loss]) 
    exp_model = exp.fit(smoothing_level=args.smoothing) 
    result = exp_model.fittedvalues
    plt.plot(result, label=loss)

# Show plot
plt.legend(loc="upper right")
plt.grid()
plt.show()
