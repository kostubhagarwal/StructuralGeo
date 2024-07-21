""" Collection of classes for implementing metaball blobs in a geological model. """

import numpy as np
from typing import List
from .geoprocess import Deposition

class Ball:
    """ A single metaball object with a given origin, radius, and goo factor. Base building class for Blob"""
    def __init__(self, origin, radius, goo_factor=1.):
        self.origin = np.array(origin)
        self.radius = radius
        self.goo_factor = goo_factor
    
    def potential(self, points):
        # Calculate the distance from the points to the ball's origin
        distances = np.sum((points - self.origin)**2, axis=1)
        # Avoid division by zero
        distances = np.maximum(distances, 1e-6)
        return (self.radius / distances)**self.goo_factor
    
class BallListGenerator:
    """ A generator class for metaballs. Generates a list of Ball objects with random parameters.
    
    Parameters:
    ----------------
    step_range (tuple): The range of uniform sampled step sizes for the metaballs.
    rad_range (tuple): The range of uniform sampled radii for the metaballs. Radius is distance to potential of 1.
    goo_range (tuple): The range of uniform sampled goo factors for the metaballs.
    """
    def __init__(self, step_range, rad_range, goo_range):
        self.step_range = step_range
        self.rad_range = rad_range
        self.goo_range = goo_range
        
    def generate(self, n_balls, origin):
        """ Generate a list of n Ball objects with random parameters starting at seeded origin. """
        balls = []
        current_point = np.array(origin, dtype=float)
        for _ in range(n_balls):
            radius = np.random.uniform(*self.rad_range)
            goo_factor = np.random.uniform(*self.goo_range)
            balls.append(Ball(current_point, radius, goo_factor))
            
            # Generate the next point with stretching
            direction = np.random.normal(size=3)
            direction /= np.linalg.norm(direction)
            step = direction * np.random.uniform(*self.step_range)
            current_point += step
            
        return balls
    
class MetaBall(Deposition):
    """ 
    A Blob geological process that modifies points within a specified potential range.

    Parameters:
    ----------------
    balls (list): A list of Ball objects defining the metaball.
    threshold (float): The threshold potential below which points will be relabeled.
    value (int): The value to assign to points below the threshold potential.
    """
    def __init__(self, balls: List[Ball], threshold, value):
        self.balls = balls
        self.threshold = threshold
        self.value = value

    def __str__(self):
        return (f"Metaball: threshold {self.threshold:.1f}, value {self.value:.1f}, "
                f"with {len(self.balls)} balls.")
    
    def run(self, xyz, data):
        # Compute the net potential for each point in xyz
        potentials = np.zeros(xyz.shape[0])
        for ball in self.balls:
            potentials += ball.potential(xyz)
        
        # Apply the threshold and relabel points
        mask = potentials > self.threshold
        data[mask] = self.value
        
        # Return the unchanged xyz and the potentially modified data
        return xyz, data   