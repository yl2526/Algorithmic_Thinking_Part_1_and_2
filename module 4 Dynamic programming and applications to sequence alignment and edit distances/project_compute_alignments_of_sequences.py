'''
Project 4
Computing alignments of sequences
'''

# import poc_grid

def build_scoring_matrix(alphabet, diag_score, off_diag_score, dash_score): 
    '''
    Takes as input a set of characters alphabet and three scores diag_score, 
    off_diag_score, and dash_score. The function returns a dictionary of 
    dictionaries whose entries are indexed by pairs of characters in alphabet 
    plus '-'. The score for any entry indexed by one or more dashes is dash_score.
    The score for the remaining diagonal entries is diag_score. Finally, the score 
    for the remaining off-diagonal entries is off_diag_score.
    '''
    result = {}
    letters = set([letter for letter in alphabet])
    letters.add('-')
    for row in letters:
        row_dict = {}
        for column in letters:
            if column == row:
                if column == '-':
                    row_dict[column] = dash_score #-float('inf')
                else:
                    row_dict[column] = diag_score
            else:
                if '-' in [row, column]:
                    row_dict[column] = dash_score
                else:   
                    row_dict[column] = off_diag_score
        result[row] = row_dict
    return result

def compute_alignment_matrix(seq_x, seq_y, scoring_matrix, global_flag): 
    '''
    Takes as input two sequences seq_x and seq_y whose elements share 
    a common alphabet with the scoring matrix scoring_matrix. The function
    computes and returns the alignment matrix for seq_x and seq_y as 
    described in the Homework. If global_flag is True, each entry of 
    the alignment matrix is computed using the method described in Question 8
    of the Homework. If global_flag is False, each entry is computed using 
    the method described in Question 12 of the Homework.
    '''
    rows = range(len(seq_x)+1)
    columns = range(len(seq_y)+1)
    alignment = [[0 for _ in columns] for _ in rows]
    if global_flag:
        for row_index in rows[1:]:
            alignment[row_index][0] = alignment[row_index-1][0] + \
                                      scoring_matrix[seq_x[row_index-1]]['-']
        for col_index in columns[1:]:
            alignment[0][col_index] = alignment[0][col_index-1] + \
                                      scoring_matrix['-'][seq_y[col_index-1]] 
    for row_index in rows[1:]:
        for col_index in columns[1:]:
            check = max(alignment[row_index-1][col_index-1] \
                            + scoring_matrix[seq_x[row_index-1]][seq_y[col_index-1]],
                        alignment[row_index-1][col_index] \
                            + scoring_matrix[seq_x[row_index-1]]['-'],
                        alignment[row_index][col_index-1] \
                        + scoring_matrix['-'][seq_y[col_index-1]]
                       )
            if (not global_flag) and (check < 0):
                alignment[row_index][col_index] = 0
            else:
                alignment[row_index][col_index] = check
    return alignment

def compute_global_alignment(seq_x, seq_y, scoring_matrix, alignment_matrix): 
    '''
    Takes as input two sequences seq_x and seq_y whose elements share a common
    alphabet with the scoring matrix scoring_matrix. This function computes a 
    global alignment of seq_x and seq_y using the global alignment matrix 
    alignment_matrix.
    The function returns a tuple of the form (score, align_x, align_y) where 
    score is the score of the global alignment align_x and align_y. Note that 
    align_x and align_y should have the same length and may include the padding 
    character '-'.
    '''
    x_remain, y_remain = len(seq_x), len(seq_y)
    x_align, y_align = '', ''
    score = alignment_matrix[x_remain][y_remain]
    while x_remain * y_remain:
        cur_align = alignment_matrix[x_remain][y_remain]
        if cur_align == (alignment_matrix[x_remain-1][y_remain-1] 
                         + scoring_matrix[seq_x[x_remain-1]][seq_y[y_remain-1]]):
            x_align = seq_x[x_remain-1] + x_align
            y_align = seq_y[y_remain-1] + y_align
            x_remain -= 1
            y_remain -= 1
        elif cur_align == (alignment_matrix[x_remain-1][y_remain] 
                           + scoring_matrix[seq_x[x_remain-1]]['-']):
            x_align = seq_x[x_remain-1] + x_align
            y_align = '-' + y_align
            x_remain -= 1
        else:
            x_align = '-' + x_align
            y_align = seq_y[y_remain-1] + y_align
            y_remain -= 1
    while x_remain:
        x_align = seq_x[x_remain-1] + x_align
        y_align = '-' + y_align
        x_remain -= 1
    while y_remain:
        x_align = '-' + x_align
        y_align = seq_y[y_remain-1] + y_align
        y_remain -= 1
    return (score, x_align, y_align)

def compute_local_alignment(seq_x, seq_y, scoring_matrix, alignment_matrix):
    '''
    This function computes a local alignment of seq_x and seq_y using 
    the local alignment matrix alignment_matrix.
    '''
    x_align, y_align = '', ''
    
    row_max, col_index = zip(*[(max(column), column.index(max(column))) 
                               for column in alignment_matrix ])
    score = max(row_max)
    x_remain = row_max.index(max(row_max))
    y_remain = col_index[row_max.index(max(row_max))]
    score = alignment_matrix[x_remain][y_remain]
    
    while x_remain * y_remain:
        cur_align = alignment_matrix[x_remain][y_remain]
        if cur_align == 0:
            break
        if cur_align == (alignment_matrix[x_remain-1][y_remain-1] 
                         + scoring_matrix[seq_x[x_remain-1]][seq_y[y_remain-1]]):
            x_align = seq_x[x_remain-1] + x_align
            y_align = seq_y[y_remain-1] + y_align
            x_remain -= 1
            y_remain -= 1
        elif cur_align == (alignment_matrix[x_remain-1][y_remain] 
                           + scoring_matrix[seq_x[x_remain-1]]['-']):
            x_align = seq_x[x_remain-1] + x_align
            y_align = '-' + y_align
            x_remain -= 1
        else:
            x_align = '-' + x_align
            y_align = seq_y[y_remain-1] + y_align
            y_remain -= 1
    return (score, x_align, y_align)


'''
import poc_simpletest

function_to_test = [build_scoring_matrix, 
                    compute_alignment_matrix,
                    compute_global_alignment,
                    compute_local_alignment
                   ]

def run_suite(function):
    """
    test the functions
    """

    suite = poc_simpletest.TestSuite()
    function = function_to_test[0]
    print function
    
    result = {'A': {'A': 6, 'C': 2, '-': -4, 'T': 2, 'G': 2}, 
              'C': {'A': 2, 'C': 6, '-': -4, 'T': 2, 'G': 2}, 
              '-': {'A': -4, 'C': -4, '-': -4, 'T': -4, 'G': -4}, 
              'T': {'A': 2, 'C': 2, '-': -4, 'T': 6, 'G': 2}, 
              'G': {'A': 2, 'C': 2, '-': -4, 'T': 2, 'G': 6}}
    suite.run_test(function(set(['A', 'C', 'T', 'G']), 6, 2, -4), result)
    
    suite.report_results()

    suite = poc_simpletest.TestSuite()
    function = function_to_test[1]
    print function
    
    score = {'A': {'A': 6, 'C': 2, '-': -4, 'T': 2, 'G': 2}, 
             'C': {'A': 2, 'C': 6, '-': -4, 'T': 2, 'G': 2}, 
             '-': {'A': -4, 'C': -4, '-': -4, 'T': -4, 'G': -4}, 
             'T': {'A': 2, 'C': 2, '-': -4, 'T': 6, 'G': 2}, 
             'G': {'A': 2, 'C': 2, '-': -4, 'T': 2, 'G': 6}}
    suite.run_test(function('A', 'A', score, True),  [[0, -4], [-4, 6]])
    
    suite.report_results()

    suite = poc_simpletest.TestSuite()
    function = function_to_test[2]
    print function

    suite.run_test(function('', '', score, [[0]]), '')
    suite.report_results()

    suite = poc_simpletest.TestSuite()
    function = function_to_test[3]
    print function
    
    suite.run_test(function('A', 'A', score, [[0]]),  [[0, -4], [-4, 6]])
    
    suite.report_results()

    
run_suite(function_to_test)
'''


