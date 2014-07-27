import cProfile
import heatmaptest
command = """heatmaptest"""
cProfile.runctx( command, globals(), locals(), filename="../output/heatmaptest.profile" ) 
