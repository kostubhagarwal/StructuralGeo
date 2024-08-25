# -*- coding: utf-8 -*-

"""
GeoGen Geological Modeling Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GeoGen is a Python library for generating and manipulating synthetic geological models.
Basic usage:

   >>> import geogen
   >>> model = geogen.GeoModel(bounds=((0, 100), (0, 100), (0, 100)), resolution=(100, 100, 50))
   >>> word = geogen.word.SingleDikeWarped()
   >>> history = word.generate()
   >>> model.add_history(history)
   >>> model.compute_model(normalize=True)

You can visualize the model using PyVista or with some built-in visualization tools:

   >>> p = geogen.plot.volview(model) # Returns a PyVista plotter object
   >>> p.show()                       # Show the plotter window
   >>> geogen.plot.categorical_grid_view(model).show()  # Directly display the model

:copyright: (c) 2024 by Simon Ghyselincks.
:license: Apache 2.0, see LICENSE for more details.
"""

import structgeo.generation as gen
import structgeo.model as geo
import structgeo.plot as geovis
import structgeo.probability as rv
