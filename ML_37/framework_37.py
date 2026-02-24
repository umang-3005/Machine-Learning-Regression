import argparse
import numpy as np
import pandas as pd
import operator 

def framework(pairs, arr):
    """
    Args:
       - pairs:  a list of (cond, calc) tuples. calc() must be an executable
       - arr: a numpy array with the features in order feat_1, feat_2, ...
    
    Executes the first calc() whose cond returns True.
    Returns None if no condition matches.
    """
    targets = []

    for i in range(arr.shape[0]):
        row = arr[i]
        for cond, calc in pairs:
            if cond_eval(cond, row):
                targets.append(calc(row))
                break
        
    return targets


def cond_eval(condition, arr):
    """evaluate a condition
        - condition: must be a tupe of (int, string, float). The second entry must be a string from the list below, describing the operator.
          Third entry of the tuple must be a float). If condition is None, it is always evaluated to true.
        - arr: array on which the condition is evaluated
    """
    ops = {
         ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    if condition is None:
        return True
    
    op = ops[condition[1]]
    return op(arr[condition[0]], condition[2])


# ============================================================
# DISCOVERED RULES FOR target02 (NO ML AT RUNTIME)
# Uses: feat_108 (idx 108), feat_116 (idx 116),
#       feat_132 (idx 132), feat_255 (idx 255)
# Conditions on feat_132: 0.2 / 0.5 / 0.7
# ============================================================

def calc_rule1(arr):
    """Rule 1: if feat_132 <= 0.2"""
    return 1.35 * arr[108] + 1.75 * arr[116] - 0.75 * arr[255]

def calc_rule2(arr):
    """Rule 2: if 0.2 < feat_132 <= 0.5"""
    return 0.35 * arr[108] - 0.45 * arr[116] + 0.55 * arr[255]

def calc_rule3(arr):
    """Rule 3: if 0.5 < feat_132 <= 0.7"""
    return 0.15 * arr[108] + 0.85 * arr[116] - 1.95 * arr[255]

def calc_rule4(arr):
    """Rule 4: if feat_132 > 0.7"""
    return 1.85 * arr[108] - 1.75 * arr[116] - 0.75 * arr[255]


def main(args):
    # Conditions are checked in order; first True wins.
    # Because of that, we can implement if/elif/else using only "<=" checks + final None.
    condition1 = (132, "<=", 0.2)   # if feat_132 <= 0.2
    condition2 = (132, "<=", 0.5)   # elif feat_132 <= 0.5  (reached only if cond1 is False)
    condition3 = (132, "<=", 0.7)   # elif feat_132 <= 0.7  (reached only if cond1,2 are False)
    condition4 = None              # else

    pair_list = [
        (condition1, calc_rule1),
        (condition2, calc_rule2),
        (condition3, calc_rule3),
        (condition4, calc_rule4),
    ]
    
    data_array = pd.read_csv(args.eval_file_path).to_numpy(dtype=float)
    return framework(pair_list, data_array)

    
def main_example(args):

    # Example: 
    test_arr = np.ones((10,10))

    def calc1(arr):
        """square first array column"""
        return arr[0]**2

    def calc2(arr):
        """add columns 3 and 4"""
        return arr[2] + arr[3]

    condition1 = (0,">=", 0.5)
    condition2 = (8, "==", 0.0)

    predict_targets = framework([(condition1, calc1), (condition2, calc2)], test_arr)
    print(predict_targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Framework Task 2")
    parser.add_argument("--eval_file_path", required=True, help="Path to EVAL_<ID>.csv")
    args = parser.parse_args()

    target02 = main(args)
