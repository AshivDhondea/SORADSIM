readme_soradsim_example_notebooks_00.txt

Readme file for the directory example_notebooks

Author: Ashiv Dhondea
# ------------------------------------------------------------------------------ #
Created: 
16/08/17
Change log:
18/08/17: added details about notebook_003_orbitpropa_validation.ipynb
22/08/17: removed notebook_003_orbitpropa_validation.ipynb
22/08/17: added notebook_005_orbitpropa_sgp4_local_topo_visualization.ipynb 
25/08/17: added notebook_006_orbitpropa_iss_vis_mauritius.ipynb 
# ------------------------------------------------------------------------------ #
Ipython notebooks to demonstrate how to use the simulator

000. noteboook_000_intro.ipynb - corresponds to main_000_readtle.py
	demonstrates reading a Two-Line Element file.
	
001. notebook_001_orbitpropa.ipynb - 
	demonstrates orbit propagation using Cowell numerical methods.
	demonstrates plotting a satellite ground track.
	
005: notebook_005_orbitpropa_sgp4_local_topo_visualization.ipynb
	demonstrate orbit propagation using SGP4 theory.
	demonstrate plotting a satellite ground track.
	demonstrate transforming an ECEF trajectory to SEZ.
	
006: notebook_006_orbitpropa_iss_vis_mauritius.ipynb
	pretty much notebook_005_orbitpropa_sgp4_local_topo_visualization, except the observer
	is in Mauritius and that we also show how to plot a topocentric right ascension and
	declination plot and how to plot maps using basemap.
# ------------------------------------------------------------------------------ #
Make sure to copy over the required libraries into the same directory before
running an Ipython notebook.
