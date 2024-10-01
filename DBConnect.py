# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:51:21 2024

@author: prana
"""

import cx_Oracle
from scipy.sparse.linalg import lsqr

import warnings
warnings.filterwarnings('ignore')

# Oracle connection information
dsn = cx_Oracle.makedsn('localhost', '1521')

# Establish a connection
username = 'demo'
password = 'demo'
connection = cx_Oracle.connect('demo', 'demo', dsn)