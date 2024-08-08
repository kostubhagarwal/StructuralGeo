import numpy as np
import pyvista as pv
from structgeo.model import rotate

# Setup plotter
p = pv.Plotter(off_screen=True)

xy_plane = pv.Plane(center=(0, 0, 0), direction=(0.,0,1), i_size=8, j_size=10, i_resolution=4, j_resolution=4,)
p.add_mesh(pv.Sphere(radius=.3, center=xy_plane.center), color='red')  # Center point
p.add_mesh(xy_plane, color='lightgreen', opacity = .5, show_edges=True)
# p.add_floor()
northing_arrow = pv.Arrow(start=xy_plane.center, direction=(0, 1, 0), scale=8, tip_length=.20, tip_radius=0.03, shaft_radius=0.01,)
p.add_mesh(northing_arrow, color='black', label='Northing Vector')
p.add_point_labels((-3.,-3,4), ["Feature Plane"], italic=False, bold=True, show_points=False, font_size=32, text_color='black',
                   shape_color='grey', shape='rounded_rect', fill_shape=True, margin=3, shape_opacity=.1,)
p.add_point_labels((3.,-1,0), ["Horizon Plane"], italic=False, bold=True, show_points=False, font_size=32, text_color='black',
                   shape_color='grey', shape='rounded_rect', fill_shape=True, margin=3, shape_opacity=.1,)

# Add arrow for strike
strike = 25

strike_vec = rotate([0,0,1], -strike*np.pi/180) @ [0,1,0]
strike_arrow = pv.Arrow(start=xy_plane.center, direction=strike_vec,  scale=5, tip_length=.25, tip_radius=0.05, shaft_radius=0.02)
p.add_mesh(strike_arrow, color='green', label='Strike Vector')


dip =  50
rake = 65
n_vec = rotate(strike_vec, dip*np.pi/180) @ [0,0,1]
trans_plane = pv.Plane(center=(0, 0, 0), direction=n_vec, i_size=10, j_size=10)
trans_plane.rotate_vector(vector=n_vec, angle=rake, point=(0.,0,0))
p.add_mesh(trans_plane, color='lightblue', opacity = .5, show_edges=True)

# Compute and display intersection line
line_direction = np.cross([0, 0, 1], n_vec)
line = pv.Line(-8*(line_direction), 8*line_direction, )  # Adjust length as needed
p.add_mesh(line, color='black', line_width=5, )

dip_vec = rotate([0,0,1], -strike*np.pi/180) @ [1,0,0]
dip_vec = rotate(strike_vec, dip*np.pi/180) @ dip_vec
p.add_mesh(pv.Arrow(start=xy_plane.center, direction=dip_vec,  scale=5, tip_length=.25, tip_radius=0.05, shaft_radius=0.02), color='orange', label='Dip Vector')

p.add_mesh(pv.Arrow(start=xy_plane.center, direction=n_vec,  scale=5, tip_length=.25, tip_radius=0.05, shaft_radius=0.02), color='blue', label='Normal Vector')

p.add_legend(bcolor='white', border=True, size=[0.4, 0.4], loc = 'center right')

# Add axes with increased size
_ = p.add_axes(line_width=6, labels_off=False, color='black',               
                shaft_length=1,
                tip_length=.3,
                ambient=0.2,
                label_size=(.4, 0.24),)

p.show_bounds(
    grid=False, 
    location='outer', 
    ticks='outside',
    font_size=12,
    show_zlabels=False,  # Do not show z labels
    show_xlabels=False,
    show_ylabels=False,
    xtitle='Easting', 
    ytitle='Northing',  
    ztitle=''  # No title for z-axis
)

p.view_vector([1,-1,0.6])

# 
# Get the current camera position
position, focal_point, view_up = p.camera_position

# Update the camera position to translate the view
new_position = [position[0]+5, position[1]+5, position[2]-2]
new_focal_point = [focal_point[0]+5, focal_point[1]+5, focal_point[2]]
p.camera_position = [new_position, new_focal_point, view_up]
# Increase the camera distance to zoom out
p.camera.zoom(.8)

p.screenshot("strike-dip-rake.png", scale=1, transparent_background=True)