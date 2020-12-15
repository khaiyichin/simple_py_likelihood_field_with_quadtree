from utils import LikelihoodField
import numpy as np

# Map point cloud in units of meters
# map_pcl = \
#   [[0.0, 0.0],
#    [1.0, 0.0],
#    [0.0, 1.0],
#    [1.0, 1.0]]

map_pcl = \
  [[-3.0, 3.0],
   [3.0, 3.0],
   [-3.0, -3.0],
   [3.0, -3.0],
   [0.0, 0.0]]

# map_pcl = \
#   [[-1.0, -1.0],
#    [1.0, 1.0]]

# map_pcl = \
#   [[0.0, 0.0]]

map_resolution = 0.5
   
map_sigma = 0.6 # in meters

if __name__ == '__main__':  

  field = LikelihoodField(map_pcl, map_resolution, map_sigma)
  
  field.display_field()