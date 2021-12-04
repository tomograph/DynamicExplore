import math
import torch
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from scipy import interpolate
import pickle

class circle_select():
    def __init__(self,im):
        self.im = im
        self.selected_points = []
        self.fig,self.ax = plt.subplots(figsize=(8,8))
        self.ax.imshow(self.im.copy(), cmap="gray", extent=[-self.im.shape[1]/2., self.im.shape[1]/2., -self.im.shape[0]/2., self.im.shape[0]/2. ])
        self.ka = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        disconnect_button = widgets.Button(description="Disconnect mpl")
        display(disconnect_button)
        disconnect_button.on_click(self.disconnect_mpl)
        self.circle = []
        self.center = (0.0,0.0)
        self.radius = 0.0

    def circle_from_points(self, angular_resolution=50):#x,y,z are complex numbers
        x,y,z = self.selected_points[:3]
        w = z-x
        w /= y-x
        c = (x-y)*(w-abs(w)**2)/2j/w.imag-x
        r = abs(c+x)
        self.circle = torch.stack([-c.real+r*torch.cos(torch.linspace(0, 2.*math.pi, angular_resolution + 1)[:-1]),-c.imag+r*torch.sin(torch.linspace(0, 2.*math.pi, angular_resolution + 1)[:-1])], axis=1)
        self.center = [-c.real, -c.imag]
        self.radius = r
        
    def disconnect_mpl(self,_):
        self.fig.canvas.mpl_disconnect(self.ka)

    def onclick(self, event):
        self.selected_points.append(complex(event.xdata,event.ydata))
        plt.clf()
        plt.imshow(self.im, cmap="gray", extent=[-self.im.shape[1]/2., self.im.shape[1]/2., -self.im.shape[0]/2., self.im.shape[0]/2. ])
        
        x = [p.real for p in self.selected_points]
        y = [p.imag for p in self.selected_points]

        if len(self.selected_points)>2:
            self.circle_from_points()
            plt.scatter(self.circle[:,0], self.circle[:,1], color="b")
            self.fig.canvas.mpl_disconnect(self.ka)
        plt.scatter(x,y, color='r')
        plt.draw()
        
class points_select():
    def __init__(self,im1, im2):
        self.im1 = im1
        self.im2 = im2
        self.source = []
        self.target = []
        self.fig,self.ax = plt.subplots(1,2,figsize=(9,6))
        self.ax[0].imshow(self.im1.copy(), cmap="gray", origin="lower")#, extent=[-self.im1.shape[1]/2., self.im1.shape[1]/2., -self.im1.shape[0]/2., self.im1.shape[0]/2. ])
        self.ax[0].set_title("Source")
        self.ax[1].imshow(self.im2.copy(), cmap="gray", origin="lower")#, extent=[-self.im1.shape[1]/2., self.im1.shape[1]/2., -self.im1.shape[0]/2., self.im1.shape[0]/2. ])
        self.ax[1].set_title("Target")
        self.ka = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        disconnect_button = widgets.Button(description="Disconnect mpl")
        display(disconnect_button)
        disconnect_button.on_click(self.disconnect_mpl)
        
    def disconnect_mpl(self,_):
        self.fig.canvas.mpl_disconnect(self.ka)

    def onclick(self, event):
        ax = event.inaxes
        ax.scatter(event.xdata, event.ydata)
        if ax.title.get_text() == "Source":
            self.source.append([event.xdata, event.ydata])
        else:
            self.target.append([event.xdata, event.ydata])
        
        return
    
class spline_select():
    def __init__(self,im):
        self.im = im
        self.selected_points = []
        self.fig,self.ax = plt.subplots(figsize=(8,8))
        self.ax.imshow(self.im.copy(), cmap="gray", extent=[-self.im.shape[1]/2., self.im.shape[1]/2., -self.im.shape[0]/2., self.im.shape[0]/2. ])
        self.ka = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        disconnect_button = widgets.Button(description="Disconnect mpl")
        display(disconnect_button)
        disconnect_button.on_click(self.disconnect_mpl)
        self.spline = []
        
    def spline_from_points(self,angular_resolution=50):
        tck,u     = interpolate.splprep( np.array(self.selected_points[:3]).transpose(), k = 2 )
        xnew,ynew = interpolate.splev( np.linspace( 0, 1, 50 ), tck,der = 0)
        self.spline = np.stack((xnew,ynew), axis=1)
        
    def disconnect_mpl(self,_):
        self.fig.canvas.mpl_disconnect(self.ka)

    def onclick(self, event):
        print(event)
        self.ax.scatter(event.xdata, event.ydata, color = "r")
        self.selected_points.append([event.xdata, event.ydata])
        #plt.clf()
        #plt.imshow(self.im, cmap="gray", extent=[-self.im.shape[1]/2., self.im.shape[1]/2., -self.im.shape[0]/2., self.im.shape[0]/2. ])
        
        #x = self.selected_points[:,0]
        #y = self.selected_points[:,1]

        if len(self.selected_points)>2:
            self.spline_from_points()
            self.ax.scatter(self.spline[:,0], self.spline[:,1], color="b")
        #    self.fig.canvas.mpl_disconnect(self.ka)
        #plt.scatter(x,y, color='r')
        #plt.draw()

#class to draw shape
class shape():
    def __inti__(self, shape_generator):
        self.shape_generator = shape_generator
        
    def draw(self, ax, resolution=50):
        points = self.shape_generator(self, resolution)        
        ax.scatter(points[:,0], points[:,1])
        
        
#inherit from shape
class shape():
    def __init__(self, points):
        self.points = points
        
    def draw(self, ax, resolution=50):
        shape_points = self.shape_generator(resolution)
        ax.scatter(shape_points[:,0], shape_points[:,1], color="b")
    
    def shape_generator(self, resolution):
        return self.points
        
class circle(shape):
    def __init__(self, points):
        p = [complex(point[0], point[1]) for point in points]
        super().__init__(p)
        x,y,z = self.points#x,y,z are complex numbers
        w = z-x
        w /= y-x
        self.center = (x-y)*(w-abs(w)**2)/2j/w.imag-x
        self.radius = abs(self.center+x)
        
    def shape_generator(self, resolution=50):
        return np.stack((-self.center.real+self.radius*np.cos(np.linspace(0, 2.*np.pi, resolution + 1)[:-1]),-self.center.imag+self.radius*np.sin(np.linspace(0, 2.*np.pi, resolution + 1)[:-1])), axis=1)
    

class spline(shape):
    def __init__(self, points):
        super().__init__(points)
        self.tck, u = interpolate.splprep( np.array(self.points).transpose(), k = 2 )
    
    def shape_generator(self, resolution=50):
        xnew,ynew = interpolate.splev( np.linspace( 0, 1, resolution ), self.tck,der = 0)
        return np.stack((xnew,ynew), axis=1)

class annotater():
    def __init__(self, volume):
        self.volume = volume
        self.fig,self.ax = plt.subplots(figsize=(6,6))
        
        #button to change slice
        slice_button = widgets.Button(description="Next")
        slice_button.on_click(self.change_slice)
        
        prev_button = widgets.Button(description="Previous")
        prev_button.on_click(self.prev_slice)
        
        #button to change slice
        copy_button = widgets.Button(description="Copy")
        copy_button.on_click(self.copy)
        
        
        #canvas for clicking
        self.ka = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        disconnect_button = widgets.Button(description="Disconnect mpl")
        disconnect_button.on_click(self.disconnect_mpl)
        
        save_button = widgets.Button(description="Save")
        save_button.on_click(self.save)
        
        load_button = widgets.Button(description="Load")
        load_button.on_click(self.load)
        
        #Add a circle
        circle_button = widgets.Button(description="Add circle")
        circle_button.on_click(self.add_circle)
        
        #Add a spline
        spline_button = widgets.Button(description="Add spline")
        spline_button.on_click(self.add_spline)
        
        #Add a spline
        clear_button = widgets.Button(description="Remove last")
        clear_button.on_click(self.remove_last)
     
        
        display(slice_button)
        display(circle_button)
        display(spline_button)
        display(disconnect_button)
        display(clear_button)
        display(save_button)
        display(load_button)
        display(prev_button)
        display(copy_button)

        #initialize
        self.slice = 0
        self.selected_points = []
        self.shapes = [[] for i in range(volume.shape[0])]
        
        self.draw_all()
        
    def load(self,_):
        with open('curves.pkl', 'rb') as inp:
            #data = pickle.load(inp)
            #self.slice = data.slice
            #self.selected_points = data.selected_points
            self.shapes = pickle.load(inp)#.shapes
        self.draw_all()

    def disconnect_mpl(self,_):
        self.fig.canvas.mpl_disconnect(self.ka)
        
    def remove_last(self,_):
        self.selected_points.clear()
        self.shapes[self.slice].pop()
        self.draw_all()
        
    def copy(self,_):
        for shp in self.shapes[self.slice-1]:
            self.shapes[self.slice].append(shp)
        self.draw_all()
    
    def add_circle(self,_):
        c = circle(self.selected_points[:3])
        self.selected_points.clear()
        self.shapes[self.slice].append(c)
        self.draw_all()
        
    def add_spline(self,_):
        spl = spline(self.selected_points[:3])
        self.selected_points.clear()
        self.shapes[self.slice].append(spl)
        self.draw_all()
        
    def onclick(self, event):
        self.selected_points.append([event.xdata, event.ydata])
        self.draw_all()
    
    def change_slice(self,_):
        self.slice += 1
        self.selected_points = []
        self.draw_all()
    
    def prev_slice(self,_):
        self.slice -= 1
        self.selected_points = []
        self.draw_all()
    
    def draw_all(self):
        self.ax.clear()
        self.ax.imshow(self.volume[self.slice,:,:], cmap="gray")
        if len(self.selected_points) > 0:
            p = np.array(self.selected_points)
            self.ax.scatter(p[:,0], p[:,1], color = "r")
        for shp in self.shapes[self.slice]:
            shp.draw(self.ax)
        plt.draw()
    
    def save(self,_):
        with open("curves.pkl", 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self.shapes, outp, pickle.HIGHEST_PROTOCOL)

