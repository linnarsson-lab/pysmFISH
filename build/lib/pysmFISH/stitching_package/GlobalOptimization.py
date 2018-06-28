import numpy as np
import sklearn.linear_model as linmod
import logging
# import matplotlib.pyplot as plt

class GlobalOptimization:
    """
    Use linear regression to find the global transitions that fit
    the pairwise transitions best

    Variables:
    ----------
    self.logger         -- logger instance
    self.global_trans   -- 2D numpy array containg the global
                        translation for each tile. (Or empty list before
                        running performOptimization())
    """


    def __init__(self):
        """Initialize logger and an array to save global transitions."""
        self.logger = logging.getLogger(__name__)
        self.global_trans = []              # array, shape: (nr_tiles, nr_dim)

    def performOptimization(self, tile_set, contig_tuples, P, covs,
                            nr_tiles, nr_dim):
        """Use linear regression to find the global transition

        Fills up self.global_trans; a numpy array with shape (nr_tiles,
        nr_dim).

        Parameters:
        -----------
        tile_set: np.array
            Array filled with ones that has the same shape as tile set
        contig_tuples: list
            List of tuples denoting which tiles are contingent to each other.
        P: np.array
            1D numpy array containing pairwise alignment y and x coordinates 
            (and z-coordinates when applicable) for each neighbouring pair of
            tiles, array should be 2 * len(contig_typles) f for 2D data or 
            3 * len(contig_typles) for 3D data.
        covs: np.array
            Covariance for each pairwise alignment in P, array should be 
            len(contig_typles).
        nr_tiles: int
            The number of tiles in the dataset
        nr_dim: int 
            The number of dimensions the image.
        """
        self.logger.info("Calculating global transitions...")

        Z_xlen = tile_set.shape[0] * tile_set.shape[1] * nr_dim

        # Prepare covariance to function as weight
        weights = np.zeros(len(P))

        # Build Design matrix, P = ZQ * Unknown
        # Order of building: Y, X, Z
        ZQ = np.zeros((len(contig_tuples) * nr_dim, Z_xlen))
        for i, (a, b) in enumerate(contig_tuples):
            # Y row:
            Z = np.zeros((Z_xlen))
            Z[nr_dim * a:nr_dim * a + 1] = -1
            Z[nr_dim * b:nr_dim * b + 1] = 1
            ZQ[i * nr_dim, :] = Z
            # X row
            Z = np.zeros((Z_xlen))
            Z[nr_dim * a + 1:nr_dim * a + 2] = -1
            Z[nr_dim * b + 1:nr_dim * b + 2] = 1
            ZQ[i * nr_dim + 1, :] = Z
            # Z row:
            if nr_dim == 3:
                Z = np.zeros((Z_xlen))
                Z[nr_dim*a + 2:nr_dim * a + 3] = -1
                Z[nr_dim*b + 2:nr_dim * b + 3] = 1
                ZQ[i * nr_dim + 2,:] = Z
            # Prepare covariance to function as weight
            if np.isnan(covs[i]):
                weights[i * nr_dim:i * nr_dim + nr_dim] = 0.0
            else:
                weights[i * nr_dim:i * nr_dim + nr_dim] = covs[i]
        self.logger.debug("weights: {}".format(weights))
        #self.logger.debug("ZQ: {}".format(ZQ))
        # plt.figure()
        # plt.imshow(ZQ, interpolation='none')
        # plt.show()
        # Find unknown in P = ZQ * Unknown through linear regression
        lrg = linmod.LinearRegression(fit_intercept=False)
        lrg.fit(ZQ, P)
        #lrg.fit(ZQ, P, weights)

        # Get coordinates with respect to zero point (tile (0,0))
        self.logger.debug("Nr of tiles: {}, nr of results lrg: {}"
                            .format(nr_tiles, len(lrg.coef_)))
        self.global_trans = lrg.coef_.reshape((nr_tiles, nr_dim))
        self.global_trans = -1 * (-lrg.coef_.reshape((nr_tiles, nr_dim)) \
                            + lrg.coef_.reshape((nr_tiles, nr_dim))[0:1, :])
        self.logger.info("Global transition calculated, global trans:\n{}"
                            .format(self.global_trans))
