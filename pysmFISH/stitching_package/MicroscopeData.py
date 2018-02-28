import numpy as np
import numpy.ma as ma
import logging

#Own imports
from . import inout

class MicroscopeData:
    """Retrieves and stores the tile set and corner coordinates of each
    tile from raw coordinate input.

    Parameters:
    -----------

    y_flip: bool
        It is assumed that the microscope
        coordinates have their origin in right top of the image.
        During the stitching we use the left, top as the origin.
        Therefore the x-coordinates will be inverted by the
        normalization. But not the y-coords.
        The y_flip variable is designed for the cases where the
        microscope sequence is inverted in the y-direction. When
        set to True the y-coordinates will also be inverted
        before determining the tile set.
    x_coords: np.array
        Array of x coordinates, will be loaded
        as raw microscope coordinates in um by init() and
        converted to pixels and normalized to start at zero by
        normalize_coords().
    y_coords: np.array
        Array of y coordinates, will be loaded
        as raw microscope coordinates in um by init() and
        converted to pixels and normalized to start at zero by
        normalize_coords().
    z_coords: np.array
        Array of z coordinates, will be loaded
        as raw microscope coordinates in um by init() and
        converted to pixels and normalized to start at zero by
        normalize_coords().
    tile_nr: np.array
        The numbers of the tiles as found in coord_data.
    tile_set: np.array
        Array with the same shape as the
        tile set, values indicate the index at which the
        corresponing tile is found in tiles and tile_nr.
    running_av: float
        The estimated overlap between two
        neighbouring tiles in pixels, adapted for the tiles we
        already know the placement and overlap of.
    logger: logger
        logger instance
    """

    def __init__(self, coord_data, y_flip, nr_dim):
        """ Create looger, read-in the coordinates and tile numbers,
        create empty tile set.

        The tile numbers refer to the number in the file name of the
        image, they identify which image file belongs to which tile.

        Parameters:
        -----------

        coord_data: dict
            Dictionary with for each tile a list with
            x, y and z coordinate, where the key is the tile
            number.
            utils.experimental_metadata_parser() outputs such a
            dictionary when reading a file with microscope
            coordinates.
        y_flip: bool
            Designed for the cases where the
            microscope sequence is inverted in the
            y-direction. When set to True the y-coordinates will
            be inverted before determining the tile set. The
            x-direction is assumed to be always inverted.
        nr_dim: int
            Valid values: 2 and 3. The number of
            dimension the loaded tiles and stitched image will have.
        """
        #Create logger
        self.logger = logging.getLogger(__name__)

        #Set flag for y flip:
        self.y_flip = y_flip

        #Extract coordinates
        self.x_coords   = []
        self.y_coords   = []
        self.z_coords   = []
        self.tile_nr    = []
        # Do not substract 1 anymore to compensate for difference in
        # filenames and image numbers in coordinates file
        self.logger.debug('Reading coord_data:')
        for key, row in coord_data.items():
            self.logger.debug('key: {} row: {}'.format(key, row))
            self.x_coords.append(row[0])
            self.y_coords.append(row[1])
            self.z_coords.append(row[2])
            self.tile_nr.append(int(key))
        #Make sure everything is np arrays
        self.x_coords   = np.array(self.x_coords)
        self.y_coords   = np.array(self.y_coords)
        self.z_coords   = np.array(self.z_coords)
        self.tile_nr    = np.array(self.tile_nr)
        self.logger.debug(("Parsed microscope data:\n tile_nr: {} \n "
                          + "x: {} \n y: {}")
                          .format(self.tile_nr, self.x_coords,
                                  self.y_coords))
        #self.logger.debug("x, y coord and tile nr: {} {} {}"
        #                        .format(self.x_coords, self.y_coords, self.tile_nr))
        #Init tile set
        self.tile_set = []


    def __getstate__(self):
        """Causes objects to be pickled without the logger attribute"""
        class_dict = dict(self.__dict__)
        del class_dict['logger']
        return class_dict

    def __setstate__(self, state):
        """Restores the logger atribute when unpickling the object"""
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)
        # Recreate the logger file, that has not been pickled
        self.logger = logging.getLogger(__name__)

    def normalize_coords(self, pixel_size):
        """Normalize the coordinates and put the origin in upper left
        corner.
        Takes list with absolute index, x, y values from microscope
        and normalizes, converts and inverts them to get the coordinates
        in pixels with the origin in left upper corner.

        NB:In the new microscope setup the origin is in the bottom right 
        corner.


        Parameters:
        -----------

        pixel_size: float
            The size of one pixel in the microscope
            image, used to convert the microscope coordinates to pixels.
        """
        self.logger.info("Normalizing coordinates")
        # Convert to pixels
        self.x_coords /= pixel_size
        self.y_coords /= pixel_size
        # Commented out because in the new system the origin is in the bottom right
        # Flip the coordinate system to get tiles in right order (0,0 is left top)

        # OLD SYSTEM

        # self.x_coords *= -1
        # if self.y_flip:
        #     self.y_coords *= -1
        # # Normalization to get postive coordinates
        # x_min = np.amin(self.x_coords)
        # self.x_coords -= x_min
        # y_min = np.amin(self.y_coords)
        # self.y_coords -= y_min

        # self.x_coords *=1
        # self.y_coords *=1
        # if self.y_flip:
        #     self.y_coords *= -1
        
        # # NEW SYSTEM

        # Normalization to get postive coordinates
        x_min = np.amin(self.x_coords)
        self.x_coords -= x_min
        y_min = np.amin(self.y_coords)
        self.y_coords -= y_min

        x_max=np.amax(self.x_coords)
        self.x_coords -= x_max
        self.x_coords=np.abs(self.x_coords)

        y_max=np.amax(self.y_coords)
        self.y_coords -= y_max
        self.y_coords=np.abs(self.y_coords)


        self.logger.debug(('Normalized microscope data:\n tile_nr: {} '
                          + '\n x: {} \n y: {}')
                          .format(self.tile_nr, self.x_coords,
                                  self.y_coords))


    def make_tile_set(self, est_x_tol, nr_pixels, row_tol = None):
        """Based on the coordinates find a tile set.

        Use the coordinates to produce a tile set that has the shape
        of the grid the tiles should be placed on and for each tile in
        this grid gives the index for the tile_nr, x_coords and
        y_coords.
        Plots the coordinates when plot_avaible == True in inout.

        Parameters:
        -----------

        est_x_tol: int
            The estimated difference in the x
            direction between the corners of two
            neighbouring tiles in pixels
        nr_pixels: int
            Needed to estimate the distance between two separate tiles
        row_tol: int
            Row tolerance in pixels: The distance
            between y coordinates above which a tile is
            considered to belong to a new row. Default is None,
            which leads to 0.5 * nr_pixels.
        """
        self.logger.info("Making tile set")
        # If row_tolerance in not passed; set the row tolerance
        # according to number of pixels:
        if row_tol is None:
            row_tol = 0.5 * nr_pixels
        self.logger.debug("Row tolereance: {}".format(row_tol))
        # Pre-sort y-coords
        coord_inds          = np.arange(0,len(self.tile_nr))
        sorting_inds        = np.argsort(self.y_coords)
        sorted_coord_inds   = coord_inds[sorting_inds]
        sorted_y            = self.y_coords[sorting_inds]

        cur_row = [sorted_coord_inds[0]]
        # Sort into rows, according to y-coordinates:
        self.logger.debug("Finding rows...")
        for i in range(1,len(sorted_y)):
            # Check the difference with the neighbour
            if abs(sorted_y[i - 1] - sorted_y[i]) > row_tol:
                self.tile_set.append(list(cur_row))
                self.logger.debug("Added row: {} ".format(cur_row))
                cur_row = [sorted_coord_inds[i]]
            else:
                cur_row.append(sorted_coord_inds[i])
        # Dump leftover in last row of tile_set
        self.tile_set.append(list(cur_row))
        self.logger.debug('Added last row: {}'.format(cur_row))

        # Sort each row according to x-coordinates
        # Initialize
        self.logger.debug("Sorting within rows...")
        self.running_av  = est_x_tol
        x_max       = max(self.x_coords) + nr_pixels
        nr_col      = int(np.around(x_max / est_x_tol))
        self.logger.debug(("Estimated x-size of final picture: {}, "
                          + "Number of expected collumns in tile "
                          + "set: {}")
                          .format(x_max, nr_col))
        av_counter  = 1
        # Sort each row in tile_set
        for i in range(len(self.tile_set)):
            # Convert to np-array
            self.tile_set[i] = np.array(self.tile_set[i])
            # Sort row
            sorting_inds = np.argsort(self.x_coords[self.tile_set[i]])
            self.logger.debug(("Row {}. "
                              + "Sorted x-coordinates:\n {}")
                              .format(i, self.x_coords[self.tile_set[
                                i]][sorting_inds]))

            # Local variable for easy adjustmet of sorted tiles
            sorted_tiles = self.tile_set[i][sorting_inds]
            #self.logger.debug("Current sorted tiles {}".format(
            # sorted_tiles))

            # Make local copy with 0 appended, for easy adjustment of sorted x coordinates
            sorted_x = np.concatenate(([0.0], self.x_coords[self.tile_set[i]][sorting_inds]))

            # Check for missing tiles:
            # Get distance between the tiles
            diff_x = abs(sorted_x[1:] - sorted_x[:-1])
            self.logger.debug("Row {}. Distance between tiles:\n {}"
                              .format(i, diff_x))
            # Update the average distance between directly adjacent tiles
            neighbouring_tiles = np.argwhere(diff_x[1:] < nr_pixels)
            if neighbouring_tiles.any():
                #self.logger.debug("running av calculated with {}".format(diff_x[1:][neighbouring_tiles]))
                #self.logger.debug("running av calculated with {}".format(np.mean(diff_x[1:][neighbouring_tiles])))
                av_counter = 2
                self.running_av = (self.running_av
                                    + np.mean(diff_x[1:][neighbouring_tiles])) \
                                    / av_counter
            self.logger.debug(("Row {}. Running average of estimated "
                              + "overlap: {}")
                              .format(i, self.running_av))

            # Find missing tiles (ignore the first tile, this one is checked later)
            missing_tiles = np.argwhere(diff_x[1:] > nr_pixels)
            missing_tiles = [tile[0] + 1 for tile in missing_tiles]
            self.logger.debug("Row {}. Missing tiles: {}".format(i,
                missing_tiles))

            # Check how many tiles are missing and insert substituting value: -1
            for diff_ind in missing_tiles:
                #self.logger.debug("Difference {}".format(diff_x[diff_ind]))
                nr_missing_tiles = int(np.around(diff_x[diff_ind] / self.running_av) - 1)
                ins_value = np.full(nr_missing_tiles, -1, dtype = int)
                #self.logger.debug("inserting missing tile at index {}".format(diff_ind))
                sorted_tiles = np.insert(sorted_tiles, diff_ind, ins_value)
                #self.logger.debug("inserting missing tile, tiles: {}".format(sorted_tiles))

            # Check for missing tiles at the start insert substituting value: -1
            if diff_x[0] > row_tol:
                nr_missing_tiles = int(np.around(diff_x[0] / self.running_av))
                ins_value = np.full(nr_missing_tiles, -1, dtype = int)
                sorted_tiles = np.insert(sorted_tiles, 0, ins_value)

            # Check if this row is as long as the rest, if it not, assume
            # that tiles are missing at the end
            # and insert substituting value: -1
            if len(sorted_tiles) < nr_col:
                nr_missing_tiles = nr_col - len(sorted_tiles)
                ins_value = np.full(nr_missing_tiles, -1, dtype = int)
                sorted_tiles = np.append(sorted_tiles, ins_value)

            # Mask the substituting value
            sorted_tiles = ma.masked_equal(sorted_tiles, -1)
            self.logger.debug("Row {}. Current sorted tiles {}".format(
                i, sorted_tiles))
            # Add to tile set:
            self.tile_set[i]  = sorted_tiles


        # Mask the tile set
        self.tile_set = ma.array(self.tile_set)
        #self.tile_set = np.array(self.tile_set)

        # Logging and plot to check
        self.logger.info("Tile set:\n {}".format(self.tile_set))
        self.logger.info("Tile set shape: {}".format(
            self.tile_set.shape))
        self.logger.info("Tile numbers:\n {}".format( self.tile_nr))

        inout.plot_coordinates(self, invert_yaxis=False)


    def check_tile_set(self, est_x_tol, nr_pixels, row_tol = None):
        """Check if the estimated overlap between tiles is close enough
        to the running average.

        Parameters:
        -----------

        est_x_tol: int
            The distance between two tiles in the
            x-direction as estimated before reading all coordinates.
        nr_pixels: int
            The the width of the tile in pixels.
        row_tol: int
            The distance between y coordinates
            above which a tile is considered to belong to a
            new row, this value is passed directly to
            make_tile_set. (default = None)
        """
        x_max       = max(self.x_coords)
        nr_col      = int(np.around(x_max / est_x_tol))
        if ((abs(est_x_tol - self.running_av) > 1000)
            or not(self.tile_set.shape[1] == nr_col)):
                self.make_tile_set(self.running_av, nr_pixels,
                                    row_tol = row_tol)
