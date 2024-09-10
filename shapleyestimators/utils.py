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
        if filename is not None:
            with open(filename, 'a') as f:
                f.write(to_print + '\n')
    if filename is not None:
        with(open(filename, 'a')) as f:
            f.write('\\bottomrule\n')
            f.write('\\end{tabular}')

def one_big_table(results, filename):
    # Each column is a dataset
    # There are five groups of rows: one for each method
    # Each group has 4 rows: mean, 1st quartile, 2nd quartile, 3rd quartile
    table = np.zeros((5*4, len(results)))
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



# Combine the two tables

# Table 1
# \begin{tabular}{lllll}
#   \toprule
#   \textbf{Method} & \textbf{Mean} & \textbf{1st Quartile} & \textbf{2nd Quartile} & \textbf{3rd Quartile} \\ \midrule 
# KernelSHAP & 2.89e-06 & 8.63e-07 & 1.30e-06 & 2.35e-06\\
# KernelSHAP Paired & \cellcolor{bronze!60}2.54e-07 & \cellcolor{gold!60}3.36e-08 & \cellcolor{bronze!60}1.09e-07 & \cellcolor{bronze!60}2.10e-07\\
# Official KernelSHAP & \cellcolor{silver!60}1.61e-07 & \cellcolor{silver!60}4.52e-08 & \cellcolor{silver!60}8.54e-08 & \cellcolor{gold!60}1.70e-07\\
# LeverageSHAP & 3.05e-06 & 5.87e-07 & 1.25e-06 & 2.52e-06\\
# LeverageSHAP Paired & \cellcolor{gold!60}1.32e-07 & \cellcolor{bronze!60}5.07e-08 & \cellcolor{gold!60}8.22e-08 & \cellcolor{silver!60}1.96e-07\\
# \bottomrule
# \end{tabular}

# Table 2
# \begin{tabular}{lllll}
#   \toprule
#   \textbf{Method} & \textbf{Mean} & \textbf{1st Quartile} & \textbf{2nd Quartile} & \textbf{3rd Quartile} \\ \midrule 
# KernelSHAP & 1.01e+00 & \cellcolor{gold!60}1.00e+00 & 1.01e+00 & 1.01e+00\\
# KernelSHAP Paired & \cellcolor{gold!60}1.00e+00 & \cellcolor{gold!60}1.00e+00 & \cellcolor{gold!60}1.00e+00 & \cellcolor{gold!60}1.00e+00\\
# Official KernelSHAP & \cellcolor{gold!60}1.00e+00 & \cellcolor{gold!60}1.00e+00 & \cellcolor{gold!60}1.00e+00 & \cellcolor{gold!60}1.00e+00\\
# LeverageSHAP & 1.01e+00 & \cellcolor{gold!60}1.00e+00 & \cellcolor{gold!60}1.00e+00 & 1.01e+00\\
# LeverageSHAP Paired & \cellcolor{gold!60}1.00e+00 & \cellcolor{gold!60}1.00e+00 & \cellcolor{gold!60}1.00e+00 & \cellcolor{gold!60}1.00e+00\\
# \bottomrule
# \end{tabular}

# Combine Table 1 and Table 2 into a single table with the following formatting

#\begin{tabular} {lccccc||cccccc}
#    \toprule
#    & \multicolumn{5}{c}{\textbf{Calibration (Negative Log-Likelihood)}} & \multicolumn{5}{c}{\textbf{Consistency}} \\ 
#    \textbf{Approach} & \textbf{ACS} & \textbf{Adult} & \textbf{Bank} & \textbf{COMPAS} & \textbf{German} & \textbf{ACS} & \textbf{Adult} & \textbf{Bank} & \textbf{COMPAS} & \textbf{German} \\ \midrule
#    \textit{Ensemble} & \cellcolor{bronze!30}\footnotesize{1.1 \smallerpm 0.04} & \footnotesize{0.9 \smallerpm 0.02} & \cellcolor{bronze!30}\footnotesize{0.5 \smallerpm 0.02} & \cellcolor{bronze!30}\footnotesize{1.8 \smallerpm 0.07} & \cellcolor{bronze!30}\footnotesize{0.97 \smallerpm 0.12} & \cellcolor{gold!30}\footnotesize{0.08 \smallerpm 0.00} & \cellcolor{gold!30}\footnotesize{0.08 \smallerpm 0.01} & \cellcolor{gold!30}\footnotesize{0.09 \smallerpm 0.01} & \cellcolor{gold!30}\footnotesize{0.06 \smallerpm 0.00} & \cellcolor{gold!30}\footnotesize{0.062 \smallerpm 0.00} \\ 
#    \emph{Selective Ens.} & \cellcolor{bronze!30}\footnotesize{1.1 \smallerpm 0.05} & \cellcolor{bronze!30}\footnotesize{0.88 \smallerpm 0.02} & \cellcolor{bronze!30}\footnotesize{0.5 \smallerpm 0.02} & \cellcolor{bronze!30}\footnotesize{1.8 \smallerpm 0.08} & \footnotesize{1.0 \smallerpm 0.16} & \footnotesize{0.45 \smallerpm 0.01} & \footnotesize{0.45 \smallerpm 0.01} & \footnotesize{0.45 \smallerpm 0.01} & \footnotesize{0.44 \smallerpm 0.01} & \footnotesize{0.40 \smallerpm 0.02} \\ 
#    \textit{(In)cons. Ens.} & \cellcolor{silver!30}\footnotesize{1.0 \smallerpm 0.04} & \cellcolor{silver!30}\footnotesize{0.82 \smallerpm 0.03} & \cellcolor{silver!30}\footnotesize{0.42 \smallerpm 0.02} & \cellcolor{silver!30}\footnotesize{1.5 \smallerpm 0.07} & \cellcolor{silver!30}\footnotesize{0.82 \smallerpm 0.13} & \cellcolor{bronze!30}\footnotesize{0.26 \smallerpm 0.01} & \cellcolor{bronze!30}\footnotesize{0.26 \smallerpm 0.01} & \cellcolor{bronze!30}\footnotesize{0.25 \smallerpm 0.01} & \cellcolor{bronze!30}\footnotesize{0.25 \smallerpm 0.01} & \cellcolor{bronze!30}\footnotesize{0.21\smallerpm 0.01} \\ 
#    \textit{Binom. NLL} & \cellcolor{gold!30}\footnotesize{0.4 \smallerpm 0.01} & \cellcolor{gold!30}\footnotesize{0.31 \smallerpm 0.0} & \cellcolor{gold!30}\footnotesize{0.2 \smallerpm 0.0} & \cellcolor{gold!30}\footnotesize{0.6 \smallerpm 0.01} & \cellcolor{gold!30}\footnotesize{0.5 \smallerpm 0.04} & \cellcolor{silver!30}\footnotesize{0.10 \smallerpm 0.01} & \cellcolor{silver!30}\footnotesize{0.13 \smallerpm 0.01} & \cellcolor{silver!30}\footnotesize{0.12 \smallerpm 0.01} & \cellcolor{silver!30}\footnotesize{0.07 \smallerpm 0.00} & \cellcolor{silver!30}\footnotesize{0.08 \smallerpm 0.01} \\ 
#    \bottomrule
#    \end{tabular}}

#def combine_tables(filename1, filename2, output_filename):
#    with open(filename1, 'r') as f:
#        lines1 = f.readlines()
#    with open(filename2, 'r') as f:
#        lines2 = f.readlines()
#    with open(output_filename, 'w') as f:
#        f.write('\\begin{tabular} {lccccc||cccccc}\n')
#        f.write('    \\toprule\n')
#        f.write('    & \\multicolumn{5}{c}{\\textbf{$\ell_2$ Error}} & \\multicolumn{5}{c}{\\textbf{Objective Error}} \\\\ \n')
#        colnames = '& \\textbf{Mean} & \\textbf{1st Quartile} & \\textbf{2nd Quartile} & \\textbf{3rd Quartile}'
#        f.write('    \\textbf{Approach} ' + colnames + ' & ' + colnames + ' \\\\ \\midrule \n')
#        for i in range(3, len(lines1)):
#            f.write(lines1[i].strip().replace('\\\\', '') + ' & ' + lines2[i].strip() + '\n')
#        f.write('    \\bottomrule\n')
#        f.write('    \\end{tabular}')