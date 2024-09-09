import numpy as np
from tabulate import tabulate

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
        print(to_print)
        if filename is not None:
            with open(filename, 'a') as f:
                f.write(to_print + '\n')
    if filename is not None:
        with(open(filename, 'a')) as f:
            f.write('\\bottomrule\n')
            f.write('\\end{tabular}')