#!/usr/bin/env python3
import sys
from subprocess import PIPE, Popen

import numpy as np
import visdom
from console_progressbar import ProgressBar

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


class Visualizer:
    def __init__(self, opt):
        self.vis = visdom.Visdom(
            server="http://localhost", port=8097, env="main"
        )
        if not self.vis.check_connection():
            self.create_visdom_connections()

        self.pb = ProgressBar(total=opt.num_epochs)

    def create_visdom_connections(self):
        cmd = f"{sys.executable} -m visdom.server -p 8097 &>/dev/null &"
        print(
            "\n\nCould not connect to visdom server.\n"
            "Trying to start a server..."
        )
        print(f"Command: {cmd}")
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def print_progress_bar(self, epoch, counter_ratio):
        self.pb.print_progress_bar(epoch + counter_ratio)

    def plot_current_losses(self, epoch, counter_ratio, losses):
        if not hasattr(self, "loss_plot_data"):
            self.loss_plot_data = {
                "X": [],
                "Y": [],
                "title": "Loss over time",
                "legend": list(losses.keys()),
                "ylabel": "loss",
            }
        self.loss_plot_data["X"].append(epoch + counter_ratio)
        self.loss_plot_data["Y"].append(
            [losses[k] for k in self.loss_plot_data["legend"]]
        )
        self._plot("loss_plot_data", 0)

    def plot_current_accuracy(self, epoch, accuracies):
        if not hasattr(self, "accuracy_plot_data"):
            self.accuracy_plot_data = {
                "X": [],
                "Y": [],
                "title": "Accuracy over time",
                "legend": list(accuracies.keys()),
                "ylabel": "accuracy",
            }
        self.accuracy_plot_data["X"].append(epoch)
        self.accuracy_plot_data["Y"].append(
            [accuracies[k] for k in self.accuracy_plot_data["legend"]]
        )
        self._plot("accuracy_plot_data", 1)

    def _plot(self, attr, win_id):
        if not hasattr(self, attr):
            raise Exception("Attribute %s does not exist", attr)
        plot_data = getattr(self, attr)
        try:
            self.vis.line(
                X=np.stack(
                    [np.array(plot_data["X"])] * len(plot_data["legend"]),
                    1,
                ),
                Y=np.array(plot_data["Y"]),
                opts={
                    "title": plot_data["title"],
                    "legend": plot_data["legend"],
                    "xlabel": "epoch",
                    "ylabel": plot_data["ylabel"],
                },
                win=win_id,
            )
        except VisdomExceptionBase:
            self.create_visdom_connection()
