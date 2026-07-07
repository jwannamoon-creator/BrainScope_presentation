from nilearn import datasets, plotting
import matplotlib.pyplot as plt

fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

plotting.plot_surf(
    fsaverage.pial_left,
    bg_map=fsaverage.sulc_left,
    hemi="left",
    view="lateral",
    title="FreeSurfer fsaverage5 - Left Hemisphere"
)

plt.show()