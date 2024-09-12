import numpy as np
from tabulate import tabulate
import math

def ith_combination(pool, r, index):
    # Function written by ChatGPT
    """
    Compute the index-th combination (0-based) in lexicographic order
    without generating all previous combinations.
    """
    n = len(pool)
    combination = []
    elements_left = n
    k = r
    start = 0
    
    for i in range(r):
        # Find the largest value for the first element in the combination
        # that allows completing the remaining k-1 elements
        for j in range(start, elements_left):
            count = math.comb(elements_left - j - 1, k - 1)
            if index < count:
                combination.append(pool[j])
                k -= 1
                start = j + 1
                break
            index -= count
    
    return tuple(combination)

def combination_generator(gen, n, s, num_samples):
    """
    Generate num_samples random combinations of s elements from a pool num_samples of size n in two settings:
    1. If the number of combinations is small (converting to an int does NOT cause an overflow error), randomly sample num_samples integers without replacement and generate the corresponding combinations on the fly with ith_combination.
    2. If the number of combinations is large (converting to an int DOES cause an overflow error), randomly sample num_samples combinations directly with replacement.
    """
    num_combos = math.comb(n, s)
    try:
        indices = gen.choice(num_combos, num_samples, replace=False)
        for i in indices:
            yield ith_combination(range(n), s, i)
    except OverflowError:
        for _ in range(num_samples):
            yield gen.choice(n, s, replace=False)

def fancy_round(x, precision=3):
    return float(np.format_float_positional(x, precision=precision, unique=False, fractional=False, trim='k'))

def benchmark_table(results, filename=None, print_md=True, include_color=True):
    table = []
    for method in results:
        row = [method]
        values = results[method][list(results[method].keys())[0]]
        mean = np.mean(values)
        median = np.median(values)
        upper = np.percentile(values, 75)
        lower = np.percentile(values, 25)
        to_add = [mean, lower, median, upper]
        row += [fancy_round(x) for x in to_add]
        table.append(row)

    if print_md:
        print(tabulate(table, headers=['Method', 'Mean', '1st Quartile', '2nd Quartile', '3rd Quartile'], tablefmt="github"))    

    cols = []
    for i in range(1,len(table[0])):
        vals = [row[i] for row in table]
        cols += [sorted(vals)]
    if filename is not None:
        with open(filename, 'w') as f:
            f.write('\\begin{tabular}{lllll}\n')
            f.write('  \\toprule\n')
            f.write('  \\textbf{Method} & \\textbf{Mean} & \\textbf{1st Quartile} & \\textbf{2nd Quartile} & \\textbf{3rd Quartile} \\\\ \\midrule \n')

    for row in table:
        print_row = [row[0]]
        for idx in range(1, len(row)):
            color = ''
            if include_color:
                if row[idx] == cols[idx-1][0]:
                    color = '\\cellcolor{gold!60}'
                elif row[idx] == cols[idx-1][1]:
                    color = '\\cellcolor{silver!60}'
                elif row[idx] == cols[idx-1][2]:
                    color = '\\cellcolor{bronze!60}'
            val = "{:.2e}".format(row[idx])
            print_row.append(f'{color}{val}')

        to_print = ' & '.join(print_row) + r'\\'
        if filename is not None:
            with open(filename, 'a') as f:
                f.write(to_print + '\n')
    if filename is not None:
        with(open(filename, 'a')) as f:
            f.write('\\bottomrule\n')
            f.write('\\end{tabular}')

def one_big_table(results, filename):
    # Each column is a dataset
    # There are several groups of rows: one for each method
    # Each group has 4 rows: mean, 1st quartile, 2nd quartile, 3rd quartile
    num_methods = len(results[list(results.keys())[0]])
    table = np.zeros((num_methods*4, len(results)))
    for i, dataset in enumerate(results):
        for j, method in enumerate(results[dataset]):
            values = np.array(results[dataset][method][list(results[dataset][method].keys())[0]])
            mean = np.mean(values)
            median = np.median(values)
            upper = np.percentile(values, 75)
            lower = np.percentile(values, 25)
            to_add = np.array([mean, lower, median, upper])
            table[j*4:(j+1)*4, i] = to_add
    with open(filename, 'w') as f:
        f.write('\\resizebox{\\linewidth}{!}{ \n')
        f.write('\\begin{tabular} {l'+ 'c'*len(results) + '}\n')
        f.write('\\toprule\n')
        f.write(' & ' + ' & '.join([f'\\textbf{{{dataset}}}' for dataset in results]) + ' \\\\ \n')
        f.write('\\midrule\n')
        i = 0
        for method in results[dataset]:
            f.write('\\addlinespace[1ex] \n')
            f.write(f'\\textbf{{{method}}}' + ' & ' * len(results) + ' \\\\ \n')
            for metric in ['Mean', '1st Quartile', '2nd Quartile', '3rd Quartile']:
                row = [] 
                for j in range(len(results)):
                    # Color the best, second, third values with gold, silver, bronze
                    # Select every 4th row

                    relevant_col = sorted(table[(i%4)::4,j])
                    color = ''
                    if table[i,j] == relevant_col[0]:
                        color = '\\cellcolor{gold!60}'
                    elif table[i,j] == relevant_col[1]:
                        color = '\\cellcolor{silver!60}'
                    elif table[i,j] == relevant_col[2]:
                        color = '\\cellcolor{bronze!60}'
                    row += [color + str(fancy_round(table[i,j]))]
                start = '\\hspace{7pt}' + metric + ' & '
                f.write(start + ' & '.join([str(x) for x in row]) + ' \\\\ \n')
                i += 1
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}}')            
        
def one_big_table_old(results, filename, include_color=True):
    # Each row is a dataset
    # There are five groups of columns: one for each method
    # Each group has 4 columns: mean, 1st quartile, 2nd quartile, 3rd quartile
    table = []
    for dataset in results:
        row = [dataset]
        for method in results[dataset]:
            values = np.array(results[dataset][method][list(results[dataset][method].keys())[0]])
            mean = np.mean(values)
            median = np.median(values)
            upper = np.percentile(values, 75)
            lower = np.percentile(values, 25)
            to_add = [mean, lower, median, upper]
            row += [fancy_round(x) for x in to_add]
        table.append(row)

    with open(filename, 'w') as f:
        f.write('\\resizebox{\\linewidth}{!}{ \n')
        f.write('\\begin{tabular} {lcccc||cccc||cccc||cccc||cccc||cccc}\n')
        f.write('\\toprule\n')
        f.write('& \\multicolumn{4}{c}{\\textbf{Kernel SHAP}} & \\multicolumn{4}{c}{\\textbf{Kernel SHAP Paired}} & \\multicolumn{4}{c}{\\textbf{Official Kernel SHAP}} & \\multicolumn{4}{c}{\\textbf{Leverage SHAP}} & \\multicolumn{4}{c}{\\textbf{Leverage SHAP Paired}} \\\\ \n')
        cols = ['Mean', '1st', '2nd', '3rd']
        colnames = ' & '.join(['\\textbf{' + col + '}' for col in cols])
        f.write(f'\\textbf{{Approach}} & {colnames} & {colnames} & {colnames} & {colnames} & {colnames} \\\\ \\midrule \n')
        for row in table:
            print_row = [row[0]]
            for i in range(1, len(row)):
                color = ''
                if include_color:
                    vals = [row[j] for j in range(1, len(row))]
                    if row[i] == min(vals):
                        color = '\\cellcolor{gold!60}'
                    elif row[i] == max(vals):
                        color = '\\cellcolor{silver!60}'
                val = "{:.2e}".format(row[i])
                print_row.append(f'{color}{val}')
            to_print = ' & '.join(print_row) + r'\\'
            f.write(to_print + '\n')
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}}')
