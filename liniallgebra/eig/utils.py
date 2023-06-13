""" Misc. tools for working with
"""
from tabulate import tabulate

def monitoring_report(A, iteration, report_iteration, monitor_setting=True):
    # we "peek" into the structure of matrix A from time to time
    # to see how it looks
    if monitor_setting and (iteration % report_iteration == 0):
        print("A",iteration,"=")
        print(tabulate(A))
        print("\n")
    return
