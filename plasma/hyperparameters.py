import numpy as np
import random
import abc
import os


class Hyperparam(object):
    @abc.abstractmethod
    def choice(self):
        return 0

    def get_conf_entry(self, conf):
        el = conf
        for sub_path in self.path:
            el = el[sub_path]
        return el

    def assign_to_conf(self, conf, save_path):
        val = self.choice()
        print(" : ".join(self.path) + ": {}".format(val))
        el = conf
        for sub_path in self.path[:-1]:
            el = el[sub_path]
        el[self.path[-1]] = val

        with open(os.path.join(save_path, "changed_params.out"), "a+") as outfile:
            for el in self.path:
                outfile.write("{} : ".format(el))
            outfile.write("{}\n".format(val))


class CategoricalHyperparam(Hyperparam):
    def __init__(self, path, values):
        self.path = path
        self.values = values

    def choice(self):
        return random.choice(self.values)


class GridCategoricalHyperparam(Hyperparam):
    def __init__(self, path, values):
        self.path = path
        self.values = iter(values)

    def choice(self):
        return next(self.values)


class ContinuousHyperparam(Hyperparam):
    def __init__(self, path, lo, hi):
        self.path = path
        self.lo = lo
        self.hi = hi

    def choice(self):
        return float(np.random.uniform(self.lo, self.hi))


class LogContinuousHyperparam(Hyperparam):
    def __init__(self, path, lo, hi):
        self.path = path
        self.lo = self.to_log(lo)
        self.hi = self.to_log(hi)

    def to_log(self, num_val):
        return np.log10(num_val)

    def choice(self):
        return float(np.power(10, np.random.uniform(self.lo, self.hi)))


class IntegerHyperparam(Hyperparam):
    def __init__(self, path, lo, hi):
        self.path = path
        self.lo = lo
        self.hi = hi

    def choice(self):
        return int(np.random.random_integers(self.lo, self.hi))


class GenericHyperparam(Hyperparam):
    def __init__(self, path, choice_fn):
        self.path = path
        self.choice_fn = choice_fn

    def choice(self):
        return self.choice_fn()


class HyperparamExperiment(object):
    def __init__(self, path, conf_name="conf.yaml"):
        if not path.endswith("/"):
            path += "/"
        self.path = path
        self.finished = False
        self.success = False
        self.logs_path = path + "/epoch_train_log.txt"
        self.raw_logs_path = path[:-1] + ".out"
        self.changed_path = os.path.join(path, "changed_params.out")
        # with open(os.path.join(self.path, conf_name), "r") as yaml_file:
        #     conf = yaml.load(yaml_file, Loader=yaml.SafeLoader)
        self.name_to_monitor = "Val ROC"  # conf['callbacks']['monitor']
        self.load_data()
        self.get_changed()
        self.get_maximum()
        self.read_raw_logs()

    def __lt__(self, other):
        return self.path.__lt__(other.path)

    def get_number(self):
        try:
            ret = int(os.path.basename(self.path[:-1]))
        except:
            ret = self.path[:-1]
        return ret  # int(os.path.basename(self.path[:-1]))

    def __str__(self):
        s = "Experiment:\n"
        s += "-" * 20 + "\n"
        s += "# {}\n".format(self.get_number())
        s += "-" * 20 + "\n"
        s += self.changed
        s += "-" * 20 + "\n"
        s += "Maximum of {} at epoch {}\n".format(*self.get_maximum(False))
        s += "-" * 20 + "\n"
        return s

    def summary(self):
        s = "Finished" if self.finished else "Running"
        print(
            "# {} [{}] maximum of {} at epoch {}".format(
                self.get_number(), s, *self.get_maximum(False)
            )
        )

    def load_data(self):
        print("reading log", self.logs_path)
        if os.path.isfile(self.logs_path):
            if os.path.getsize(self.logs_path) > 0:
                self.epochs = []
                self.values = []
                self.dat = []
                with open(self.logs_path, "r") as file:
                    lines = file.readlines()
                    for line in lines[1:]:
                        l_stripped = line.strip().split()
                        self.dat.append(l_stripped)
                        self.epochs.append(float(l_stripped[0]))
                        self.values.append(float(l_stripped[3]))
                    print("loaded logs")
                    print(self.epochs)
                    print(self.values)
                    return
        self.epochs = []
        print("no logs yet")

    def get_changed(self):
        with open(self.changed_path, "r") as file:
            text = file.read()
        print("changed values: {}".format(text))
        self.changed = text
        return text

    def read_raw_logs(self):
        self.success = False
        self.finished = False
        lines = []
        if os.path.exists(self.raw_logs_path):
            with open(self.raw_logs_path, "r") as file:
                lines = file.readlines()
        if len(lines) > 1:
            if lines[-1].strip() == "done.":
                self.finished = True
                if lines[-2].strip() == "finished.":
                    self.success = True
        print("finished: {}, success: {}".format(self.finished, self.success))

    def get_maximum(self, verbose=True):
        if len(self.epochs) > 0:
            idx = np.argmax(self.values)
            s = "Finished" if self.finished else "Running"
            if verbose:
                # print(self.path)
                print(
                    "[{}] maximum of {} at epoch {}".format(
                        s, self.values[idx], self.epochs[idx]
                    )
                )
            return self.values[idx], self.epochs[idx]
        else:
            return -1, -1
