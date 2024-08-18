import numpy as np

from structgeo.model.geoprocess import DeferredParameter, Transformation

class BacktrackedPoint(DeferredParameter):
    """
    A deferred parameter used to specify a point in the final coordinate frame, which is then backtracked 
    through the mesh deformations to determine its original position in the initial coordinate frame.

    This class is useful in geological modeling when a point of interest is defined in the final (present-day) 
    configuration of the model, but needs to be traced back through all previous transformations to find its 
    corresponding position at an earlier stage of geological history.

    Parameters
    ----------
    point : tuple
        A 3D coordinate tuple (x, y, z) representing the point in the final coordinate frame.

    Methods
    -------
    compute_func(xyz, data, history, index)
        Resolves the deferred parameter by applying the inverse of all transformations that occurred 
        after the current process in the model's history, effectively backtracking the point to its original 
        position in the initial coordinate frame.
    """    
    def __init__(self, point: tuple):
        super().__init__()
        self.point = point
        
    def __str__(self):
        return f"BacktrackedPoint: {self.point}"
    
    def compute_func(self, xyz, data, history, index) -> tuple:
        point = np.atleast_2d(np.array(self.point)) # Cast tuple into 2D array for processing
        data   = np.array([np.nan]) # Dummy data to go with point
        
        # Reverse through the history events applying transformations
        future = history[index+1:]
        for event in reversed(future):
            if isinstance(event, Transformation):
                point , _ = event.run(xyz = point, data = data)
                
        point = tuple(point.flatten()) # Cast back to 3d tuple
    
        return point