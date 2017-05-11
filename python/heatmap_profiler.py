import cProfile
import heatmaptest
command = """heatmaptest"""
cProfile.runctx( command, globals(), locals(), filename="./output/heatmaptest.profile" ) 

import gp_classifier_test
command = """gp_classifier_test"""
cProfile.runctx( command, globals(), locals(), filename="./output/gp_classifier_test.profile" ) 
