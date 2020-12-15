from scipy import stats
from pyqtree import Index
import numpy as np

class LikelihoodField:

  def __init__(self, map_pointcloud2d, map_resolution, map_sigma, debug_mode=False):
    """

    Args:
    """
    if map_resolution*1000 % 1 != 0:
      print('Error: The resolution must be >= 1 mm!')
      exit()

    self.debug_mode = debug_mode

    self.map_pointcloud2d = map_pointcloud2d # assuming that map_poincloud2d is a list of x,y points
    self.map_resolution = map_resolution
    self.map_sigma = map_sigma

    # Initialize the likelihood function pdf
    self.likelihood_function_pdf = stats.norm(0, map_sigma)

    # Initialize the quadtree
    self.compute_field_bounding_box()
    self.quadtree = Index( bbox=self.bbox, max_items=1, max_depth=np.log2(self.bbox_len/self.map_resolution) )

    # Build the likelihood field
    self.build()

  def compute_field_bounding_box(self):
    """

    Returns:
      A tuple containing the minimum x, y values, the maximum x, y values, and the length of the square bounding box.
    """
    # Get dimension of map in 2-D
    min_x_dim = min(self.map_pointcloud2d, key=lambda x:x[0])[0]
    max_x_dim = max(self.map_pointcloud2d, key=lambda x:x[0])[0]
    min_y_dim = min(self.map_pointcloud2d, key=lambda x:x[1])[1]
    max_y_dim = max(self.map_pointcloud2d, key=lambda x:x[1])[1]

    self.sigma_padding = self.round_to_multiple(2.5 * self.map_sigma, self.map_resolution, 'inf') # only consider values within 3 std dev

    x_padding = (self.round_to_multiple(min_x_dim, self.map_resolution, 'inf') - min_x_dim - self.sigma_padding,
                 self.round_to_multiple(max_x_dim, self.map_resolution, 'inf') - max_x_dim + self.sigma_padding)
    y_padding = (self.round_to_multiple(min_y_dim, self.map_resolution, 'inf') - min_y_dim - self.sigma_padding,
                 self.round_to_multiple(max_y_dim, self.map_resolution, 'inf') - max_y_dim + self.sigma_padding)

    # Compute bounding boxes
    bbox_x = (min_x_dim + x_padding[0], max_x_dim + x_padding[1])
    bbox_y = (min_y_dim + y_padding[0], max_y_dim + y_padding[1])
    bbox_len = bbox_x[1] - bbox_x[0]

    self.bbox = (bbox_x[0], bbox_y[0], bbox_x[1], bbox_y[1])
    self.bbox_len = bbox_len
    cell_num_1D = int(round(self.bbox_len / self.map_resolution) + 1)
    print('Field size (length x width)\t\t\t\t=\t' + str(cell_num_1D) + ' x ' + str(cell_num_1D) + ' cells.')
    print('Field bounds (xmin, ymin, xmax, ymax)\t=\t' + str(self.bbox) + '.')
    print('Field resolution\t\t\t\t\t\t=\t' + str(self.map_resolution) + ' m.')

  def build(self):
    """Builds the likelihood field from the sample point cloud map.
    """
    self.sample_pointcloud()
    self.build_likelihood_field()

  def round_to_multiple(self, value, base, mode='r'):
    """Rounds a value to the next multiple of a specific base.
    """
    if mode == 'r': # regular mode
      return np.copysign(round(base * round(float(abs(value)) / base), 3), value)
    elif mode == 'inf': # round towards +/- infinity
      return np.copysign(round(base * np.ceil(float(abs(value)) / base), 3), value)
    elif mode == '0': # round towards zero
      return np.copysign(round(base * np.floor(float(abs(value)) / base), 3), value)

  def sample_pointcloud(self):
    """Samples the obstacles in the point cloud map.
    """

    print('\nSampling map point cloud... ', end='')

    # Populate the map_grid by going through the list of pointcloud data (should be sorted, maybe not?)
    points = self.map_pointcloud2d
    map_grid_obstacles_coords = []
    while len(points) != 0:
      
      point = points[0]

      x_sampled = self.round_to_multiple(point[0], self.map_resolution)
      y_sampled = self.round_to_multiple(point[1], self.map_resolution)

      # Store the sampled map obstacle point
      map_grid_obstacles_coords.append((x_sampled, y_sampled))

      # Pop the current point
      points.pop(0)
    
    self.map_grid_obstacles_coords = np.asarray(np.unique(map_grid_obstacles_coords, axis=0)) # filter the repeated obstacles and convert the list into numpy array

    print('Done!')

  def build_likelihood_field(self):
    """Build the likelihood field and store it in the quadtree.
    """

    print('\nBuilding likelihood field... ', end='')

    stored_likelihoods = []
    
    voxel_radius = self.sigma_padding
    
    # Iterate through list of obstacles in the grid to compute the likelihood and then populate the quadtree
    for obs_coord in self.map_grid_obstacles_coords:

      for dist_x in np.arange(-voxel_radius, voxel_radius + self.map_resolution, self.map_resolution):
          for dist_y in np.arange(-voxel_radius, voxel_radius + self.map_resolution, self.map_resolution):

            # Calculate likelihood, and the bounding boxes for that likelihood
            x_min = obs_coord[0] + dist_x - self.map_resolution/4.0
            x_max = obs_coord[0] + dist_x + self.map_resolution/4.0            
            y_min = obs_coord[1] + dist_y - self.map_resolution/4.0
            y_max = obs_coord[1] + dist_y + self.map_resolution/4.0
            bbox = (x_min, y_min, x_max, y_max)

            distance = np.sqrt( np.square(dist_x) + np.square(dist_y) )

            # Convert the likelihoods to uint8 values to save memory
            likelihood_uint8 = self.likelihood_dec2uint8(self.evaluate_cell_likelihood(distance))

            if likelihood_uint8 == 0: continue # only store non-zero values

            # Check to see if the likelihood was stored previously, and only store the higher likelihood
            stored_likelihood = [likelihood_bbox_pair for likelihood_bbox_pair in stored_likelihoods if likelihood_bbox_pair[1] == bbox]
            
            if stored_likelihood:
              # Replace the lower likelihood with the current, higher one
              if stored_likelihood[0][0] < likelihood_uint8:
                stored_likelihoods.remove(stored_likelihood[0])
                stored_likelihoods.append( (likelihood_uint8, bbox) )

            else:
              stored_likelihoods.append( (likelihood_uint8, bbox) )
            

    for likelihood_bbox_pair in stored_likelihoods:
      self.quadtree.insert(likelihood_bbox_pair[0], likelihood_bbox_pair[1])

    print('Done!')

    # Display likelihood field information (used for debugging)
    if self.debug_mode:
      
      print('Voxel radius evaluated: ' + str(voxel_radius) + ' m = ' + str(voxel_radius/self.map_resolution) + ' cells.')

      likelihood_num = len(stored_likelihoods)
      print('Number of stored likelihood values: ' + str(int(likelihood_num)))
      print('Stored likelihood values and bounding boxes:')

      for i in range(likelihood_num):
        if i == 0: print('[' + str(stored_likelihoods[i]))
        elif i == likelihood_num-1: print(' ' + str(stored_likelihoods[i]) + ']')
        else: print(' ' + str(stored_likelihoods[i]))

  def evaluate_cell_likelihood(self, distance_from_obstacle):
    """Computes the likelihood probability of a cell.
    """
    return abs(self.likelihood_function_pdf.cdf(distance_from_obstacle - self.map_resolution/2.0) - self.likelihood_function_pdf.cdf(distance_from_obstacle + self.map_resolution/2.0))

  def likelihood_dec2uint8(self, likelihood_val_dec):
    return np.uint8(likelihood_val_dec * 255)

  def likelihood_uint82dec(self, likelihood_val_uint8):
    return likelihood_val_uint8 / 255.0

  def display_field(self):
    """Displays the stored likehood field in grid form.
    """
    grid = np.asarray([[0.0 for _ in range(int(round(self.bbox_len/self.map_resolution)) + 1)] for _ in range(int(round(self.bbox_len/self.map_resolution) + 1))])
    counter = 0

    grid_ind_row = len(grid)-1

    for y in np.arange(self.bbox[1], self.bbox[3]+self.map_resolution, self.map_resolution):
      grid_ind_col = 0

      for x in np.arange(self.bbox[0], self.bbox[2]+self.map_resolution, self.map_resolution):

        val = self.quadtree.intersect( (x - self.map_resolution/4, y - self.map_resolution/4, x + self.map_resolution/4, y + self.map_resolution/4) ) # get the region centered within a cell so that it doesn't overlap other cells

        if val:
          counter += 1
          grid[grid_ind_row][grid_ind_col] = val[0] # there should be only one value in the val list
      
        grid_ind_col += 1
        
      grid_ind_row -= 1

    print('\nNumber of likelihood values populated: ' + str(int(counter)))
    print('Likelihood field: ')
    with np.printoptions(threshold=np.inf, linewidth=np.inf): print(grid)