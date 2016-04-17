'''
This script is some basic excrie for the graph operations
'''

EX_GRAPH0 = {0: set([1,2]), 1: set([]), 2: set([])}
EX_GRAPH1 = {0: set([1,4, 5]), 
             1: set([2, 6]), 
             2: set([3]), 
             3: set([0]), 
             4: set([1]), 
             5: set([2]), 
             6: set([])
            }
EX_GRAPH2 = {0: set([1, 4, 5]), 
             1: set([2, 6]),
             2: set([3, 7]), 
             3: set([7]), 
             4: set([1]), 
             5: set([2]), 
             6: set([]), 
             7: set([3]), 
             8: set([1, 2]), 9: set([0, 3, 4, 5, 6, 7])
            }

def make_complete_graph(num_nodes):
    '''
    Takes the number of nodes num_nodes and returns a 
    dictionary corresponding to a complete directed graph 
    with the specified number of nodes
    '''
    graph_dict = {}
    for key in xrange(num_nodes):
        graph_dict[key] = set(range(key) + range(key+1, num_nodes))
    return graph_dict
        
def compute_in_degrees(digraph):
    '''
    Takes a directed graph digraph (represented as a dictionary)
    and computes the in-degrees for the nodes in the graph.
    '''
    in_degree = {}
    for head, tails in digraph.items():
        in_degree[head] = in_degree.get(head, 0)
        for tail in tails:
            in_degree[tail] = in_degree.get(tail, 0) + 1
    return in_degree

def in_degree_distribution(digraph):
    '''
    Takes a directed graph digraph (represented as a dictionary) 
    and computes the unnormalized distribution of the in-degrees 
    of the graph.
    '''
    in_degree_dict = compute_in_degrees(digraph)
    distribute = {}
    for in_degree in in_degree_dict.values():
        distribute[in_degree] = distribute.get(in_degree, 0) + 1
    return distribute

'''
import poc_simpletest
function_to_test = [make_complete_graph, compute_in_degrees, in_degree_distribution]
def run_suite(function):
    """
    test the functions
    """
    suite = poc_simpletest.TestSuite()
    print function[0]
    suite.run_test(function[0](0), {})
    suite.run_test(function[0](1), {0: set([])})
    suite.run_test(function[0](3), {0: set([1, 2]), 1: set([0, 2]), 2: set([0, 1])})
    suite.report_results()
    
    suite = poc_simpletest.TestSuite()
    print function[1]
    suite.run_test(function[1](EX_GRAPH0), {0: 0, 1: 1, 2: 1})
    suite.run_test(function[1](EX_GRAPH1), {0: 1, 1: 2, 2: 2, 3: 1, 
                                            4: 1, 5: 1, 6: 1})
    suite.run_test(function[1](EX_GRAPH2), {0: 1, 1: 3, 2: 3, 3: 3, 
                                            4: 2, 5: 2, 6: 2, 7: 3, 
                                            8: 0, 9: 0})
    suite.report_results()
    
    suite = poc_simpletest.TestSuite()
    print function[2]
    suite.run_test(function[2](EX_GRAPH0), {0: 1, 1: 2})
    suite.run_test(function[2](EX_GRAPH1), {1: 5, 2: 2})
    suite.run_test(function[2](EX_GRAPH2), {0: 2, 1: 1, 2: 3, 3: 4})
    suite.report_results()
    
run_suite(function_to_test)
'''
