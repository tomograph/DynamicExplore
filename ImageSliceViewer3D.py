
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
from IPython.display import clear_output
#%config InlineBackend.close_figures=False 


#Define class for slicing a 3D volume
class ImageSliceViewer3D:
    """ 
    ImageSliceViewer3D is for viewing volumetric image slices in jupyter or
    ipython notebooks. 
    
    User can interactively change the slice plane selection for the image and 
    the slice plane being viewed. 

    Argumentss:
    Volume = 3D input image
    figsize = default(8,8), to set the size of the figure
    cmap = default('plasma'), string for the matplotlib colormap. You can find 
    more matplotlib colormaps on the following link:
    https://matplotlib.org/users/colormaps.html
    
    """
    def plot_slice(self, change):
        self.ax.clear()
        self.ax.imshow(self.volume[change['new'],:,:])
        with self.out:
            clear_output(wait=True)
            display(self.ax.figure)

    
    def __init__(self, volume):
        plt.ioff()
        self.volume = volume
        #self.figsize = figsize
        self.ax = plt.gca()
        self.ax.imshow(self.volume[0,:,:])
        self.out=widgets.Output()
        # Call to view a slice within the selected slice plane
        #out=widgets.Output()
        button=widgets.BoundedIntText(min=0, max=volume.shape[0]-1,continuous_update=False, description='Image Slice:')

        vbox=widgets.VBox(children=(self.out,button))
        display(vbox)
        button.observe(self.plot_slice, names='value')
        
        

