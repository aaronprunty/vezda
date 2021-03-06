# Copyright 2017-2020 Aaron C. Prunty
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#        
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#==============================================================================
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
from skimage import measure

#==============================================================================
# Define color class for printing to terminal
class FontColor:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

# General functions for plotting...

# custom Vezda colormaps for light/dark mode wiggle plots
vz_cwm = matplotlib.colors.LinearSegmentedColormap.from_list('', ['cyan', 'whitesmoke', 'magenta'])
vz_cbm = matplotlib.colors.LinearSegmentedColormap.from_list('', ['cyan', 'black', 'magenta'])

def get_cmap_colors(cmap_string, mode='light', fills=False):
    if cmap_string == 'native':
        if fills == True:
            return 'm', 'c'
        
        if mode == 'light':
            return vz_cwm
        else: # mode == 'dark'
            return vz_cbm
        
    elif cmap_string == 'grays':
        if fills == True:
            if mode == 'light':
                return 'black', 'None'
            else:
                return 'whitesmoke', 'None'
        else:
            return plt.get_cmap(cmap_string)
        
    elif cmap_string == 'seismic':
        if fills == True:
            return 'blue', 'red'
        else:
            return plt.get_cmap(cmap_string)
    
        
def default_params():
    '''
    Sets the default ploting parameters used in both
    image/map plots as well as wiggle plots.
    
    This function takes no arguments.
    '''
    
    plotParams = {}
    #for both image and wiggle plots
    plotParams['pltformat'] = 'pdf'
    plotParams['view_mode'] = 'light'
    
    # for image/map plots
    plotParams['isolevel'] = 0.7
    plotParams['xlabel'] = ''
    plotParams['ylabel'] = ''
    plotParams['zlabel'] = ''
    plotParams['xu'] = ''
    plotParams['yu'] = ''
    plotParams['zu'] = ''
    plotParams['image_colormap'] = 'magma'
    plotParams['wiggle_colormap'] = 'grays'
    plotParams['colorbar'] = False
    plotParams['shading'] = 'flat'
    plotParams['vmin'] = 0.0
    plotParams['vmax'] = 1.0
    plotParams['levels'] = 100
    plotParams['invert_xaxis'] = False
    plotParams['invert_yaxis'] = False
    plotParams['invert_zaxis'] = False
    plotParams['show_scatterer'] = False
    plotParams['show_sources'] = True
    plotParams['show_receivers'] = True
    
    # for wiggle plots
    plotParams['pclip'] = 1.0
    plotParams['tu'] = ''
    plotParams['au'] = ''
    plotParams['data_title'] = 'Data'
    plotParams['impulse_title'] = 'Impulse Response'
    
    # for frequency plots
    plotParams['fu'] = ''
    plotParams['fmin'] = 0
    plotParams['fmax'] = None
    plotParams['freq_title'] = 'Mean Amplitude Spectrum'
    plotParams['freq_ylabel'] = 'Amplitude'
    
    return plotParams



def setFigure(num_axes=1, mode='light', ax1_dim=2, ax2_dim=2):
    '''
    Create figure and axes objects styled for daytime viewing
    
    num_axes: number of axes objects to return (either 1 or 2)
    ax(1,2)_dim: number of dimensions in the axis object (i.e., 2D or 3D plot)
    ax(1,2)_grid: boolean (True of False)
    '''
    
    plt.style.use('ggplot')
    plt.rc('axes', titlesize=13)
    
    if num_axes == 1:
        fig = plt.figure()
        
        if ax1_dim == 2:
            ax1 = fig.add_subplot(111)
        elif ax1_dim == 3:
            ax1 = fig.add_subplot(111, projection='3d')
            
        if mode == 'light':
            ax1.spines['left'].set_color('black')
            ax1.spines['right'].set_color('black')
            ax1.spines['top'].set_color('black')
            ax1.spines['bottom'].set_color('black')
            ax1.cbaredgecolor = 'darkgray'
            ax1.alpha = 0.6
            ax1.shadecolor = 'black'
            ax1.shadealpha = 0.2
            ax1.linecolor = 'slategray'
            ax1.pointcolor = 'black'
            ax1.linewidth = 0.95
            ax1.labelcolor = '#4c4c4c'
            ax1.titlecolor = 'black'
            ax1.receivercolor = 'black'
            ax1.sourcecolor = 'black'
            ax1.activesourcecolor = 'darkcyan'
            ax1.inactivesourcecolor = 'darkgray'
            ax1.scatterercolor = 'darkgray'
            ax1.surfacecolor = 'c'
            #ax1.wiggle_cmap = vz_cwm
        
        elif mode == 'dark':
            plt.rc('grid', linestyle='solid', color='dimgray')
            plt.rc('legend', facecolor='353535')
            plt.rc('text', color='whitesmoke')
            
            fig.set_facecolor('black')
    
            ax1.tick_params(colors='555555')
            ax1.set_facecolor('525252')
            ax1.spines['left'].set_color('darkgray')
            ax1.spines['right'].set_color('darkgray')
            ax1.spines['top'].set_color('darkgray')
            ax1.spines['bottom'].set_color('darkgray')
            ax1.cbaredgecolor = 'darkgray'
            ax1.alpha = 0.6
            ax1.shadecolor = 'black'
            ax1.shadealpha = 0.5
            ax1.linecolor = 'silver'
            ax1.pointcolor = 'silver'
            ax1.linewidth = 0.95
            ax1.labelcolor = '555555'
            ax1.titlecolor = '555555'
            ax1.receivercolor = 'darkgray'
            ax1.sourcecolor = 'darkgray'
            ax1.activesourcecolor = 'c'
            ax1.inactivesourcecolor = 'dimgray'
            ax1.scatterercolor = 'lightgray'
            ax1.surfacecolor = 'c'
            #ax1.wiggle_cmap = vz_cbm
            
        return fig, ax1
            
    elif num_axes == 2:
        fig = plt.figure(figsize=plt.figaspect(0.48))
        
        if ax1_dim == 2:
            ax1 = fig.add_subplot(121)
        elif ax1_dim == 3:
            ax1 = fig.add_subplot(121, projection='3d')
            
        if ax2_dim == 2:
            ax2 = fig.add_subplot(122)
        elif ax2_dim == 3:
            ax2 = fig.add_subplot(122, projection='3d')
            
        if mode == 'light':
            ax1.spines['left'].set_color('black')
            ax1.spines['right'].set_color('black')
            ax1.spines['top'].set_color('black')
            ax1.spines['bottom'].set_color('black')
            ax1.cbaredgecolor = 'darkgray'
            ax1.alpha = 0.6
            ax1.shadecolor = 'black'
            ax1.shadealpha = 0.2
            ax1.linecolor = 'slategray'
            ax1.pointcolor = 'black'
            ax1.linewidth = 0.95
            ax1.labelcolor = '#4c4c4c'
            ax1.titlecolor = 'black'
            ax1.receivercolor = 'black'
            ax1.sourcecolor = 'black'
            ax1.activesourcecolor = 'darkcyan'
            ax1.inactivesourcecolor = 'darkgray'
            ax1.scatterercolor = 'darkgray'
            ax1.surfacecolor = 'c'
            #ax1.wiggle_cmap = vz_cwm
            
            ax2.spines['left'].set_color('black')
            ax2.spines['right'].set_color('black')
            ax2.spines['top'].set_color('black')
            ax2.spines['bottom'].set_color('black')
            ax2.cbaredgecolor = 'darkgray'
            ax2.alpha = 0.6
            ax2.shadecolor = 'black'
            ax2.shadealpha = 0.2
            ax2.linecolor = 'slategray'
            ax2.pointcolor = 'black'
            ax2.linewidth = 0.95
            ax2.labelcolor = '#4c4c4c'
            ax2.titlecolor = 'black'
            ax2.receivercolor = 'black'
            ax2.sourcecolor = 'black'
            ax2.activesourcecolor = 'darkcyan'
            ax2.inactivesourcecolor = 'darkgray'
            ax2.scatterercolor = 'darkgray'
            ax2.surfacecolor = 'c'
            #ax2.wiggle_cmap = vz_cwm
            
        
        elif mode == 'dark':
            plt.rc('grid', linestyle='solid', color='dimgray')
            plt.rc('legend', facecolor='353535')
            plt.rc('text', color='whitesmoke')
            
            fig.set_facecolor('black')
    
            ax1.tick_params(colors='555555')
            ax1.set_facecolor('525252')
            ax1.spines['left'].set_color('darkgray')
            ax1.spines['right'].set_color('darkgray')
            ax1.spines['top'].set_color('darkgray')
            ax1.spines['bottom'].set_color('darkgray')
            ax1.cbaredgecolor = 'darkgray'
            ax1.alpha = 0.6
            ax1.shadecolor = 'black'
            ax1.shadealpha = 0.5
            ax1.linecolor = 'silver'
            ax1.pointcolor = 'silver'
            ax1.linewidth = 0.95
            ax1.labelcolor = '555555'
            ax1.titlecolor = '555555'
            ax1.receivercolor = 'darkgray'
            ax1.sourcecolor = 'darkgray'
            ax1.activesourcecolor = 'c'
            ax1.inactivesourcecolor = 'dimgray'
            ax1.scatterercolor = 'lightgray'
            ax1.surfacecolor = 'c'
            #ax1.wiggle_cmap = vz_cbm
            
            ax2.tick_params(colors='555555')
            ax2.set_facecolor('525252')
            ax2.spines['left'].set_color('darkgray')
            ax2.spines['right'].set_color('darkgray')
            ax2.spines['top'].set_color('darkgray')
            ax2.spines['bottom'].set_color('darkgray')
            ax2.cbaredgecolor = 'darkgray'
            ax2.alpha = 0.6
            ax2.shadecolor = 'black'
            ax2.shadealpha = 0.5
            ax2.linecolor = 'silver'
            ax2.pointcolor = 'silver'
            ax2.linewidth = 0.95
            ax2.labelcolor = '555555'
            ax2.titlecolor = '555555'
            ax2.receivercolor = 'darkgray'
            ax2.sourcecolor = 'darkgray'
            ax2.activesourcecolor = 'c'
            ax2.inactivesourcecolor = 'dimgray'
            ax2.scatterercolor = 'lightgray'
            ax2.surfacecolor = 'c'
            #ax2.wiggle_cmap = vz_cbm
        
        return fig, ax1, ax2


#==============================================================================
# Functions for gradient shading under curves...
def zfunc(x, y, fill_color='k', alpha=1.0):
    scale = 10
    x = (x*scale).astype(int)
    y = (y*scale).astype(int)
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()

    w, h = xmax-xmin, ymax-ymin
    z = np.empty((h, w, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(fill_color)
    z[:, :, :3] = rgb

    # Build a z-alpha array which is 1 near the line and 0 at the bottom.
    img = Image.new('L', (w, h), 0)  
    draw = ImageDraw.Draw(img)
    xy = (np.column_stack([x, y]))
    xy -= xmin, ymin
    
    # Draw a blurred line using PIL
    draw.line(list(map(tuple, xy.tolist())), fill=255, width=15)
    img = img.filter(ImageFilter.GaussianBlur(radius=h//8))
    
    # Convert the PIL image to an array
    zalpha = np.asarray(img).astype(float) 
    zalpha *= alpha/zalpha.max()
    
    # make the alphas melt to zero at the bottom
    n = zalpha.shape[0] // 4
    zalpha[:n] *= np.linspace(0, 1, n)[:, None]
    z[:, :, -1] = zalpha
    
    return z


def gradient_fill(x, y, fill_color=None, ax=None, zfunc=None, scale='linear', **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    fill_color : a matplotlib color specifier (string, tuple) or None
        The color for the fill. If None, the color of the line will be used.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.
    """
    if ax is None:
        ax = plt.gca()

    if scale == 'linear':
        line, = ax.plot(x, y, color=ax.linecolor, linewidth=ax.linewidth, **kwargs)
    elif scale == 'log':
        line, = ax.semilogy(x, y, color=ax.linecolor, linewidth=ax.linewidth, **kwargs)
    
    if fill_color is None:
        fill_color = line.get_color()

    zorder = line.get_zorder()
    alpha = line.get_alpha()
    alpha = 1.0 if alpha is None else alpha

    if zfunc is None:
        h, w = 200, 1
        z = np.empty((h, w, 4), dtype=float)
        rgb = mcolors.colorConverter.to_rgb(fill_color)
        z[:, :, :3] = rgb
        z[:, :, -1] = np.linspace(0, alpha, h)[:, None]
    else:
        z = zfunc(x, y, fill_color=fill_color, alpha=alpha)
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower', zorder=zorder)

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    clip_path = patches.Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    ax.autoscale(True)
    return line, im        

#==============================================================================
# Functions for plotting waveforms...
def set_ylabel(N, coordinates, id_number, flag, plotParams):
    '''
    Sets the appropriate y-axis label according to the object being plotted
    
    N: number of points along y-axis (>= 1)
    coordinates: location of object being referenced on y-axis (only used if N == 1)
    id_number: the number of the object being plotted (e.g., 'Receiver 1')
    flag: string parameter describing the type of the object ('data', 'impulse', 'left', or 'right')
    plotParams: a dictionary of the plot parameters for styling
    '''
    
    if flag == 'data' or flag == 'impulse' or flag == 'left':
        ylabel = 'Receiver'
    elif flag == 'right':
        ylabel = 'Source'
    
    if N == 1:
        
        # get amplitude units from plotParams
        au = plotParams['au']
        
        if coordinates is not None:
            # get units for x and y axes from plotParams
            xu = plotParams['xu']
            yu = plotParams['yu']
        
            # update ylabel to also show amplitude and coordinate information
            if coordinates.shape[1] == 2:
                if au != '' and xu != '' and yu != '':
                    ylabel = 'Amplitude (%s) [%s %s @ (%0.2f %s, %0.2f %s)]' %(au, ylabel, id_number,
                                                                               coordinates[0, 0], xu,
                                                                               coordinates[0, 1], yu)
                elif au == '' and xu != '' and yu != '':
                    ylabel = 'Amplitude [%s %s @ (%0.2f %s, %0.2f %s)]' %(ylabel, id_number,
                                                                          coordinates[0, 0], xu,
                                                                          coordinates[0, 1], yu)
                elif au != '' and xu == '' and yu == '':
                    ylabel = 'Amplitude (%s) [%s %s @ (%0.2f, %0.2f)]' %(au, ylabel, id_number,
                                                                         coordinates[0, 0],
                                                                         coordinates[0, 1])
                elif au == '' and xu == '' and yu == '':
                    ylabel = 'Amplitude [%s %s @ (%0.2f, %0.2f)]' %(ylabel, id_number,
                                                                    coordinates[0, 0],
                                                                    coordinates[0, 1])
                    
            elif coordinates.shape[1] == 3:
                # get units for z axis from plotParams
                zu = plotParams['zu']
            
                if au != '' and xu != '' and yu != '' and zu != '':
                    ylabel = 'Amplitude (%s) [%s %s @ (%0.2f %s, %0.2f %s, %0.2f %s)]' %(au, ylabel, id_number,
                                                                                         coordinates[0, 0], xu,
                                                                                         coordinates[0, 1], yu,
                                                                                         coordinates[0, 2], zu)
                elif au == '' and xu != '' and yu != '' and zu != '':
                    ylabel = 'Amplitude [%s %s @ (%0.2f %s, %0.2f %s, %0.2f %s)]' %(ylabel, id_number,
                                                                                    coordinates[0, 0], xu,
                                                                                    coordinates[0, 1], yu,
                                                                                    coordinates[0, 2], zu)
                elif au != '' and xu == '' and yu == '' and zu == '':
                    ylabel = 'Amplitude (%s) [%s %s @ (%0.2f, %0.2f, %0.2f)]' %(au, ylabel, id_number,
                                                                                coordinates[0, 0],
                                                                                coordinates[0, 1],
                                                                                coordinates[0, 2])
                elif au == '' and xu == '' and yu == '' and zu == '':
                    ylabel = 'Amplitude [%s %s @ (%0.2f, %0.2f, %0.2f)]' %(ylabel, id_number,
                                                                           coordinates[0, 0],
                                                                           coordinates[0, 1],
                                                                           coordinates[0, 2])
        else:   # coordinates is None
            
            # update ylabel to also show amplitude information
            if au != '':
                ylabel = 'Amplitude (%s) [%s %s]' %(au, ylabel, id_number)
            else:
                ylabel = 'Amplitude [%s %s]' %(ylabel, id_number)
    
    return ylabel



def plotWiggles(ax, X, xvals, interval, coordinates, title, flag, plotParams):
    ax.clear()
    N = X.shape[0]
    
    id_number = interval[0]
    ylabel = set_ylabel(N, coordinates, id_number, flag, plotParams)
    ax.set_ylabel(ylabel, color=ax.labelcolor)
    
    if N > 1:
        
        if N <= 18:
            ax.set_yticks(interval)                
            ax.set_yticklabels(interval)
            plt.setp(ax.get_yticklabels(), visible=True)
            plt.setp(ax.get_yticklines(), visible=True)
            pos_fill, neg_fill = get_cmap_colors(plotParams['wiggle_colormap'], plotParams['view_mode'], fills=True)
            
            # rescale all wiggle traces by largest displacement range
            scaleFactor = np.max(np.ptp(X, axis=1))
            if scaleFactor != 0:
                X /= scaleFactor
            
            for n in range(N):
                ax.plot(xvals, interval[n] + X[n, :], color=ax.linecolor, linewidth=ax.linewidth)
                ax.fill_between(xvals, interval[n], interval[n] + X[n, :],
                                where=(interval[n] + X[n, :] > interval[n]), color=pos_fill, alpha=ax.alpha)
                ax.fill_between(xvals, interval[n], interval[n] + X[n, :],
                                where=(interval[n] + X[n, :] < interval[n]), color=neg_fill, alpha=ax.alpha)
                
        elif N > 18 and N <= 70:            
            pos_fill, neg_fill = get_cmap_colors(plotParams['wiggle_colormap'], plotParams['view_mode'], fills=True)
            
            # rescale all wiggle traces by largest displacement range
            scaleFactor = np.max(np.ptp(X, axis=1))
            if scaleFactor != 0:
                X /= scaleFactor
            
            for n in range(N):
                ax.plot(xvals, interval[n] + X[n, :], color=ax.linecolor, linewidth=ax.linewidth)
                ax.fill_between(xvals, interval[n], interval[n] + X[n, :],
                                where=(interval[n] + X[n, :] > interval[n]), color=pos_fill, alpha=ax.alpha)
                ax.fill_between(xvals, interval[n], interval[n] + X[n, :],
                                where=(interval[n] + X[n, :] < interval[n]), color=neg_fill, alpha=ax.alpha)
        
        else:
            scaleFactor = np.max(np.abs(X))   
            pclip = plotParams['pclip']
            wiggle_cmap = get_cmap_colors(plotParams['wiggle_colormap'], plotParams['view_mode'])
            #wiggle_cmap = plt.get_cmap(plotParams['wiggle_colormap'])
            ax.pcolormesh(xvals, interval, X, vmin=-scaleFactor * pclip, vmax=scaleFactor * pclip, cmap=wiggle_cmap)
                        
    else: # N == 1
        pos_fill, neg_fill = get_cmap_colors(plotParams['wiggle_colormap'], plotParams['view_mode'],fills=True)
        
        ax.yaxis.get_offset_text().set_x(-0.1)
        ax.plot(xvals, X[0, :], color=ax.linecolor, linewidth=ax.linewidth)
        ax.fill_between(xvals, 0, X[0, :], where=(X[0, :] > 0), color=pos_fill, alpha=ax.alpha)
        ax.fill_between(xvals, 0, X[0, :], where=(X[0, :] < 0), color=neg_fill, alpha=ax.alpha)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
    ax.set_title(title, color=ax.titlecolor)
    # get time units from plotParams
    tu = plotParams['tu']
    if tu != '':
        ax.set_xlabel('Time (%s)' %(tu), color=ax.labelcolor)
    else:
        ax.set_xlabel('Time', color=ax.labelcolor)
    
    ax.set_xlim([xvals[0], xvals[-1]])
        
    return ax


def plotFreqVectors(ax, volume, xvals, interval, coordinates, title, flag, plotParams):
    ax.clear()
    
    ax.set_title(title, color=ax.titlecolor)
    # get frequency units from plotParams
    fu = plotParams['fu']
    if fu != '':
        ax.set_xlabel('Frequency (%s)' %(fu), color=ax.labelcolor)
    else:
        ax.set_xlabel('Frequency', color=ax.labelcolor)
    
    N = volume.shape[0]
    id_number = interval[0]
    ylabel = set_ylabel(N, coordinates, id_number, flag, plotParams)
    ax.set_ylabel(ylabel, color=ax.labelcolor)
    if N > 1:
        # rescale all wiggle traces by largest displacement range
        scaleFactor = np.max(np.abs(volume))
        if scaleFactor != 0:
            volume /= scaleFactor
        
        wiggle_cmap = get_cmap_colors(plotParams['wiggle_colormap'], plotParams['view_mode'])
        return ax.pcolormesh(xvals, interval, volume, vmin=-1, vmax=1, cmap=wiggle_cmap)
                        
    else: # N == 1
        ax.yaxis.get_offset_text().set_x(-0.1)
                
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        return ax.stem(xvals, volume[0, :], linefmt=ax.linecolor, markerfmt='mo')


#==============================================================================
# Functions for plotting maps and images
def plotMap(ax, index, receiverPoints, sourcePoints, scatterer, flag, plotParams):
    if index is None:
        
        if sourcePoints is None:
            if receiverPoints.shape[1] == 2:
                if plotParams['show_receivers']:
                    ax.plot(receiverPoints[:, 0], receiverPoints[:, 1], 'v', color=ax.receivercolor)
        
                if scatterer is not None and plotParams['show_scatterer']:
                    ax.plot(scatterer[:, 0], scatterer[:, 1], '--', color=ax.scatterercolor)
                
                  
            elif receiverPoints.shape[1] == 3:
                if plotParams['show_receivers']:
                    ax.plot(receiverPoints[:, 0], receiverPoints[:, 1], receiverPoints[:, 2], 'v', color=ax.receivercolor)
        
                if scatterer is not None and plotParams['show_scatterer']:
                    ax.plot(scatterer[:, 0], scatterer[:, 1], scatterer[:, 2], '--', color=ax.scatterercolor)
                
                zu = plotParams['zu']
                zlabel = plotParams['zlabel']
        
                if zu != '':
                    ax.set_zlabel(zlabel + ' (%s)' %(zu), color=ax.labelcolor)
                else:
                    ax.set_zlabel(zlabel, color=ax.labelcolor)
                    
                if plotParams['invert_zaxis']:
                    ax.invert_zaxis()
            
        else:   # sourcePoints exist
            if receiverPoints.shape[1] == 2:
                if plotParams['show_receivers']:
                    ax.plot(receiverPoints[:, 0], receiverPoints[:, 1], 'v', color=ax.receivercolor)
        
                if flag == 'data' and plotParams['show_sources']:
                    ax.plot(sourcePoints[:, 0], sourcePoints[:, 1], '*', color=ax.sourcecolor)
        
                elif flag == 'impulse':
                        ax.plot(sourcePoints[:, 0], sourcePoints[:, 1], '.', color=ax.sourcecolor)
        
                if scatterer is not None and plotParams['show_scatterer']:
                    ax.plot(scatterer[:, 0], scatterer[:, 1], '--', color=ax.scatterercolor)
                
                
            elif receiverPoints.shape[1] == 3:
                if plotParams['show_receivers']:
                    ax.plot(receiverPoints[:, 0], receiverPoints[:, 1], receiverPoints[:, 2], 'v', color=ax.receivercolor)
        
                if flag == 'data' and plotParams['show_sources']:
                    ax.plot(sourcePoints[:, 0], sourcePoints[:, 1], sourcePoints[:, 2], '*', color=ax.sourcecolor)
        
                elif flag == 'impulse':
                    ax.plot(sourcePoints[:, 0], sourcePoints[:, 1], sourcePoints[:, 2], '.', color=ax.sourcecolor)
        
                if scatterer is not None and plotParams['show_scatterer']:
                    ax.plot(scatterer[:, 0], scatterer[:, 1], scatterer[:, 2], '--', color=ax.scatterercolor)
                
                zu = plotParams['zu']
                zlabel = plotParams['zlabel']
        
                if zu != '':
                    ax.set_zlabel(zlabel + ' (%s)' %(zu), color=ax.labelcolor)
                else:
                    ax.set_zlabel(zlabel, color=ax.labelcolor)
                    
                if plotParams['invert_zaxis']:
                    ax.invert_zaxis()
         
    else:
        ax.clear()
        ax.set_title('Map', color=ax.titlecolor)
        
        if sourcePoints is None:
            if receiverPoints.shape[1] == 2:
                if plotParams['show_receivers']:
                    ax.plot(receiverPoints[:, 0], receiverPoints[:, 1], 'v', color=ax.receivercolor)
        
                if scatterer is not None and plotParams['show_scatterer']:
                    ax.plot(scatterer[:, 0], scatterer[:, 1], '--', color=ax.scatterercolor)
                
                  
            elif receiverPoints.shape[1] == 3:
                if plotParams['show_receivers']:
                    ax.plot(receiverPoints[:, 0], receiverPoints[:, 1], receiverPoints[:, 2], 'v', color=ax.receivercolor)
        
                if scatterer is not None and plotParams['show_scatterer']:
                    ax.plot(scatterer[:, 0], scatterer[:, 1], scatterer[:, 2], '--', color=ax.scatterercolor)
                
                zu = plotParams['zu']
                zlabel = plotParams['zlabel']
        
                if zu != '':
                    ax.set_zlabel(zlabel + ' (%s)' %(zu), color=ax.labelcolor)
                else:
                    ax.set_zlabel(zlabel, color=ax.labelcolor)
                
                if plotParams['invert_zaxis']:
                    ax.invert_zaxis()
            
        else:
            # delete the row corresponding to the current source (plot current source separately)
            sources = np.delete(sourcePoints, index, axis=0)
            currentSource = sourcePoints[index, :]
    
            if receiverPoints.shape[1] == 2:
                if plotParams['show_receivers']:
                    ax.plot(receiverPoints[:, 0], receiverPoints[:, 1], 'v', color=ax.receivercolor)
        
                if flag == 'data' and plotParams['show_sources']:
                    ax.plot(sources[:, 0], sources[:, 1], '*', color=ax.inactivesourcecolor)
                    ax.plot(currentSource[0], currentSource[1], marker='*', markersize=12, color=ax.activesourcecolor)
        
                elif flag == 'impulse':
                    ax.plot(sources[:, 0], sources[:, 1], '.', color=ax.inactivesourcecolor)
                    ax.plot(currentSource[0], currentSource[1], marker='.', markersize=12, color=ax.activesourcecolor)
        
                if scatterer is not None and plotParams['show_scatterer']:
                    ax.plot(scatterer[:, 0], scatterer[:, 1], '--', color=ax.scatterercolor)
                
                  
            elif receiverPoints.shape[1] == 3:
                if plotParams['show_receivers']:
                    ax.plot(receiverPoints[:, 0], receiverPoints[:, 1], receiverPoints[:, 2], 'v', color=ax.receivercolor)
        
                if flag == 'data' and plotParams['show_sources']:
                    ax.plot(sources[:, 0], sources[:, 1], sources[:, 2], '*', color=ax.inactivesourcecolor)
                    ax.plot(currentSource[0], currentSource[1], currentSource[2], marker='*', markersize=12, color=ax.activesourcecolor)
        
                elif flag == 'impulse':
                    ax.plot(sources[:, 0], sources[:, 1], sources[:, 2], '.', color=ax.inactivesourcecolor)
                    ax.plot(currentSource[0], currentSource[1], currentSource[2], marker='.', markersize=12, color=ax.activesourcecolor)
        
                if scatterer is not None and plotParams['show_scatterer']:
                    ax.plot(scatterer[:, 0], scatterer[:, 1], scatterer[:, 2], '--', color=ax.scatterercolor)
                
                zu = plotParams['zu']
                zlabel = plotParams['zlabel']
        
                if zu != '':
                    ax.set_zlabel(zlabel + ' (%s)' %(zu), color=ax.labelcolor)
                else:
                    ax.set_zlabel(zlabel, color=ax.labelcolor)
                
                if plotParams['invert_zaxis']:
                    ax.invert_zaxis()
        
    ax.grid(False)
    ax.set_aspect('equal')
    
    xu = plotParams['xu']
    xlabel = plotParams['xlabel']
    
    yu = plotParams['yu']
    ylabel = plotParams['ylabel']
    
    if xu != '':
        ax.set_xlabel(xlabel + ' (%s)' %(xu), color=ax.labelcolor)
    else:
        ax.set_xlabel(xlabel, color=ax.labelcolor)
        
    if yu != '':
        ax.set_ylabel(ylabel + ' (%s)' %(yu), color=ax.labelcolor)
    else:
        ax.set_ylabel(ylabel, color=ax.labelcolor)
    
    if plotParams['invert_xaxis']:
        ax.invert_xaxis()
    
    if plotParams['invert_yaxis']:
        ax.invert_yaxis()
    
    return ax


def isosurface(volume, level, x1, x2, x3):
    
    verts, faces, normals, values = measure.marching_cubes_lewiner(volume, level=level)
    
    # Rescale coordinates of vertices to lie within x,y,z ranges
    verts[:, 0] = verts[:, 0] * (x1[-1] - x1[0]) / (np.max(verts[:, 0]) - np.min(verts[:, 0])) + x1[0]
    verts[:, 1] = verts[:, 1] * (x2[-1] - x2[0]) / (np.max(verts[:, 1]) - np.min(verts[:, 1])) + x2[0]
    verts[:, 2] = verts[:, 2] * (x3[-1] - x3[0]) / (np.max(verts[:, 2]) - np.min(verts[:, 2])) + x3[0]
    
    return verts, faces
    

def image_viewer(ax, volume, method, alpha, atol, btol, plotParams, X, Y, Z=None, tau=None):
    ax.clear()
    ax.grid(False)
    
    xu = plotParams['xu']
    xlabel = plotParams['xlabel']
    
    yu = plotParams['yu']
    ylabel= plotParams['ylabel']
    
    if xu != '':
        ax.set_xlabel(xlabel + ' (%s)' %(xu), color=ax.labelcolor)
    else:
        ax.set_xlabel(xlabel, color=ax.labelcolor)
        
    if yu != '':
        ax.set_ylabel(ylabel + ' (%s)' %(yu), color=ax.labelcolor)
    else:
        ax.set_ylabel(ylabel, color=ax.labelcolor)
    
    tu = plotParams['tu']
    
    #if method == 'lsmr':
    #    title = 'Method: LSMR\n'
    #elif method == 'lsqr':
    #    title = 'Method: LSQR\n'
    #elif method == 'svd':
    #    title = 'Method: SVD\n'
    
    if Z is None:
        image_colormap = plt.get_cmap(plotParams['image_colormap'])
        volume[volume > plotParams['vmax']] = plotParams['vmin']
        im = ax.pcolormesh(X, Y, volume, vmin=plotParams['vmin'], vmax=plotParams['vmax'],
                           cmap=image_colormap, shading=plotParams['shading'])
        if plotParams['colorbar']:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if tau is not None:
                cbar.set_label(r'$\frac{1}{\vert\vert\varphi\vert\vert}$',
                               labelpad=24, rotation=0, fontsize=18, color=ax.labelcolor)
        
        if alpha != 0.0:
            title = r'$\alpha = %0.1e$' %(alpha)
        else:
            title = r'$\alpha = %d$' %(alpha)
            
        #if method != 'svd':
        #    title += ', tol = %01.e' %(tol)
    
        #if tau is not None:
        #    if tu != '':
        #        title += r', $\tau = %0.2f$ %s' %(tau, tu)
        #    else:
        #        title += r', $\tau = %0.2f$' %(tau)
                
    else:
        x = X[:, 0, 0]
        y = Y[0, :, 0]
        z = Z[0, 0, :]
        
        isolevel = plotParams['isolevel']
        
        # Plot isosurface of support of source function in space
        verts, faces = isosurface(volume, isolevel, x, y, z)
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color=ax.surfacecolor)
                
        zu = plotParams['zu']
        zlabel = plotParams['zlabel']
        
        if zu != '':
            ax.set_zlabel(zlabel + ' (%s)' %(zu), color=ax.labelcolor)
        else:
            ax.set_zlabel(zlabel, color=ax.labelcolor)
        
        if alpha != 0:
            title += r'Isosurface @ %s [$\alpha = %0.1e$]' %(isolevel, alpha)
        else:
            title += r'Isosurface @ %s [$\alpha = %s$]' %(isolevel, alpha)
    
        if tau is not None:
            if tu != '':
                title = title[:-1] + r', $\tau = %0.2f %s]' %(tau, tu)
            else:
                title = title[:-1] + r', $\tau = %0.2f]' %(tau)
        
    ax.set_title(title, color=ax.titlecolor)
        
    return ax


def plotImage(Dict, X, Y, Z, tau, plotParams, flag, movie=False):
    # Set up a two- or three-dimensional figure
    if Z is None:      
        fig, ax = setFigure(num_axes=1, mode=plotParams['view_mode'], ax1_dim=2) 
    else:
        fig, ax = setFigure(num_axes=1, mode=plotParams['view_mode'], ax1_dim=3)
    
    Image = Dict['Image'].reshape(X.shape)
    method = Dict['method']
    alpha = Dict['alpha']
    atol = Dict['atol']
    btol = Dict['btol']
    
    if Dict['domain'] == 'time':
        image_viewer(ax, Image, method, alpha, atol, btol, plotParams, X, Y, Z, tau)
    else:
        image_viewer(ax, Image, method, alpha, atol, btol, plotParams, X, Y, Z)
        
    return fig, ax
        
        
#==============================================================================
# General functions for interactive plotting...

def remove_keymap_conflicts(new_keys_set):
    '''
    Removes pre-defined keyboard events so that interactive
    plotting with various keys can be used
    '''
    
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

#==============================================================================
# Specific functions for plotting data and test functions...
class Experiment(object):
    
    def __init__(self, data, time, receiverNumbers, receiverPoints,
                 sourceNumbers, sourcePoints, wiggleType):
        self.data = data
        self.time = time
        self.receiverNumbers = receiverNumbers
        self.receiverPoints = receiverPoints
        self.sourceNumbers = sourceNumbers
        self.sourcePoints = sourcePoints
        self.wiggleType = wiggleType
        
    
    def plot(self, scatterer, plotParams, show_map=False):        
        remove_keymap_conflicts({'left', 'right', 'up', 'down', 'save'})        
        
        if show_map:
            fig, ax1, ax2 = setFigure(num_axes=2, mode=plotParams['view_mode'],
                                      ax2_dim=self.receiverPoints.shape[1])
            
            ax1.source_index = len(self.sourceNumbers) // 2
            title = wave_title(ax1.source_index, self.sourceNumbers,
                               self.sourcePoints, self.wiggleType, plotParams)
            plotWiggles(ax1, self.data[:, :, ax1.source_index], self.time, self.receiverNumbers,
                        self.receiverPoints, title, self.wiggleType, plotParams)
        
            ax2.source_index = ax1.source_index
            plotMap(ax2, ax2.source_index, self.receiverPoints, self.sourcePoints, scatterer, self.wiggleType, plotParams)
            
            return fig, ax1, ax2
        
        else:
            fig, ax = setFigure(num_axes=1, mode=plotParams['view_mode'])
            
            ax.source_index = len(self.sourceNumbers) // 2
            title = wave_title(ax.source_index, self.sourceNumbers, self.sourcePoints, self.wiggleType, plotParams)
            plotWiggles(ax, self.data[:, :, ax.source_index], self.time, self.receiverNumbers,
                        self.receiverPoints, title, self.wiggleType, plotParams)

            return fig, ax
        

class Plotter(object):
    
    def __init__(self, original, reciprocal=None):
        self.experiments = [original, reciprocal]
        self.index = 0
        
    def plot(self, scatterer, plotParams, show_map=False):
        remove_keymap_conflicts({'r'})
        
        experiment = self.experiments[self.index]
        fig, *axes = experiment.plot(scatterer, plotParams, show_map)
        
        plt.tight_layout()
        fig.canvas.mpl_connect('key_press_event', lambda event: switch_experiments(event, self, plotParams,
                                                                                   scatterer, show_map))
        fig.canvas.mpl_connect('key_press_event', lambda event: process_key_waves(event, self, plotParams,
                                                                                   scatterer, show_map))


def switch_experiments(event, plotter, plotParams, scatterer, show_map):
    '''
    Determines how to draw the next plot based on keyboard events
    
    event: a keyboard hit, either 'left', 'right', 'up', or 'down' arrow keys
    
    Passed parameters:
    time: an array of time values over which the singular vectors are defined
    rinterval: an interval or sampling of the receivers used
    sinterval: an interval or sampling of the sources used
    receiverPoints: coordinates of the receivers
    sourcePoints: coordinates of the sources (may be None)
    scatterer: coordinates of the scatterer boundary
    show_map: Boolean value (True/False)
    flag: string parameter describing the type of the object ('data', 'impulse', 'left', or 'right')
    plotParams: a dictionary of plot parameters for styling
    '''
    experiments = plotter.experiments
    if experiments[1] is None:
        Ne = 1
    else:
        Ne = 2
    
    if event.key == 'r':
        plotter.index = (plotter.index + 1) % Ne
    
    experiment = experiments[plotter.index]
    
    if show_map:
        fig = event.canvas.figure
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
            
        title = wave_title(ax1.source_index, experiment.sourceNumbers,
                           experiment.sourcePoints, experiment.wiggleType, plotParams)
        plotWiggles(ax1, experiment.data[:, :, ax1.source_index], experiment.time,
                    experiment.receiverNumbers, experiment.receiverPoints,
                    title, experiment.wiggleType, plotParams)
        
        plotMap(ax2, ax2.source_index, experiment.receiverPoints, experiment.sourcePoints,
                scatterer, experiment.wiggleType, plotParams)
        
    else:
        fig = event.canvas.figure
        ax = fig.axes[0]
        
        title = wave_title(ax.source_index, experiment.sourceNumbers,
                           experiment.sourcePoints, experiment.wiggleType, plotParams)
        plotWiggles(ax, experiment.data[:, :, ax.source_index], experiment.time,
                    experiment.receiverNumbers, experiment.receiverPoints,
                    title, experiment.wiggleType, plotParams)
    
    fig.canvas.draw()


def process_key_waves(event, plotter, plotParams, scatterer, show_map):
    '''
    Determines how to draw the next plot based on keyboard events
    
    event: a keyboard hit, either 'left', 'right', 'up', or 'down' arrow keys
    
    Passed parameters:
    time: an array of time values over which the singular vectors are defined
    rinterval: an interval or sampling of the receivers used
    sinterval: an interval or sampling of the sources used
    receiverPoints: coordinates of the receivers
    sourcePoints: coordinates of the sources
    scatterer: coordinates of the scatterer boundary
    show_map: Boolean value (True/False)
    flag: string parameter describing the type of the object ('data', 'impulse', 'left', or 'right')
    plotParams: a dictionary of plot parameters for styling
    '''
    experiment = plotter.experiments[plotter.index]
    
    if show_map:
        fig = event.canvas.figure
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
        
        if event.key == 'left' or event.key == 'down':
            previous_wave(ax1, experiment, plotParams)
            previous_map(ax2, experiment.receiverPoints,
                         experiment.sourcePoints, scatterer,
                         experiment.wiggleType, plotParams)
            
        elif event.key == 'right' or event.key == 'up':
            next_wave(ax1, experiment, plotParams)
            next_map(ax2, experiment.receiverPoints,
                     experiment.sourcePoints, scatterer,
                     experiment.wiggleType, plotParams)
                      
    else:
        fig = event.canvas.figure
        ax = fig.axes[0]
        
        if event.key == 'left' or event.key == 'down':
            previous_wave(ax, experiment, plotParams)
        
        elif event.key == 'right' or event.key == 'up':
            next_wave(ax, experiment, plotParams)
    
    fig.canvas.draw()
         
    
def next_wave(ax, experiment, plotParams):
    Ns = len(experiment.sourceNumbers)
    ax.source_index = (ax.source_index + 1) % Ns
    title = wave_title(ax.source_index, experiment.sourceNumbers,
                       experiment.sourcePoints, experiment.wiggleType, plotParams)
    plotWiggles(ax, experiment.data[:, :, ax.source_index],
                experiment.time, experiment.receiverNumbers,
                experiment.receiverPoints, title, experiment.wiggleType, plotParams)
    

def previous_wave(ax, experiment, plotParams):
    Ns = len(experiment.sourceNumbers)
    ax.source_index = (ax.source_index - 1) % Ns  # wrap around using %
    title = wave_title(ax.source_index, experiment.sourceNumbers,
                       experiment.sourcePoints, experiment.wiggleType, plotParams)
    plotWiggles(ax, experiment.data[:, :, ax.source_index],
                experiment.time, experiment.receiverNumbers,
                experiment.receiverPoints, title, experiment.wiggleType, plotParams)


def previous_map(ax, receiverPoints, sourcePoints, scatterer, flag, plotParams):
    if sourcePoints is not None:
        Ns = sourcePoints.shape[0]
        ax.source_index = (ax.source_index - 1) % Ns  # wrap around using %
        plotMap(ax, ax.source_index, receiverPoints, sourcePoints, scatterer, flag, plotParams)
    

def next_map(ax, receiverPoints, sourcePoints, scatterer, flag, plotParams):
    if sourcePoints is not None:
        Ns = sourcePoints.shape[0]
        ax.source_index = (ax.source_index + 1) % Ns  # wrap around using %
        plotMap(ax, ax.source_index, receiverPoints, sourcePoints, scatterer, flag, plotParams)


def wave_title(index, sourceNumbers, sourcePoints, flag, plotParams):
    '''
    Creates a plot title for data arrays and test function arrays
    
    Parameters:
    index: a number indicating which source in 'interval' parameter produced the data
    interval: an integer array of source numbers
    sourcePoints: location of the sources, if known (equals 'None' otherwise)
    flag: a string parameter (either 'data' or 'impulse')
    plotParams: a dictionary of plot parameters for styling
    '''
    
    # get units for x,y axes from plotParams
    xu = plotParams['xu']
    yu = plotParams['yu']
    
    if flag == 'data':
        # get type-specific title from plotParams
        data_title = plotParams['data_title']
        
        if sourcePoints is None:
            title = '%s [Record %s/%s]' %(data_title, sourceNumbers[index], len(sourceNumbers))
        
        elif sourcePoints.shape[1] == 2:
            if  xu != '' and yu != '':
                title = '%s [Source %s @ (%0.2f %s, %0.2f %s)]' %(data_title, sourceNumbers[index],
                                                                  sourcePoints[index, 0], xu,
                                                                  sourcePoints[index, 1], yu)
            else:
                title = '%s [Source %s @ (%0.2f, %0.2f)]' %(data_title, sourceNumbers[index],
                                                            sourcePoints[index, 0],
                                                            sourcePoints[index, 1])
        elif sourcePoints.shape[1] == 3:
            # get units for z axis from plotParams
            zu = plotParams['zu']
            
            if  xu != '' and yu != '' and zu != '':
                title = '%s [Source %s @ (%0.2f %s, %0.2f %s, %0.2f %s)]' %(data_title, sourceNumbers[index],
                                                                            sourcePoints[index, 0], xu,
                                                                            sourcePoints[index, 1], yu,
                                                                            sourcePoints[index, 2], zu)
            else:
                title = '%s [Source %s @ (%0.2f, %0.2f, %0.2f)]' %(data_title, sourceNumbers[index],
                                                                   sourcePoints[index, 0],
                                                                   sourcePoints[index, 1],
                                                                   sourcePoints[index, 2])
        
    else:   # flag == 'impulse'
        # get type-specific title from plotParams
        impulse_title = plotParams['impulse_title']
        
        if sourcePoints.shape[1] == 2:
            if xu != '' and yu != '':
                title = '%s [$\\bf{z}$ @ (%0.2f %s, %0.2f %s)]' %(impulse_title,
                                                                  sourcePoints[index, 0], xu,
                                                                  sourcePoints[index, 1], yu)
            elif xu == '' and yu == '':
                title = '%s [$\\bf{z}$ @ (%0.2f, %0.2f)]' %(impulse_title,
                                                            sourcePoints[index, 0],
                                                            sourcePoints[index, 1])
                    
        elif sourcePoints.shape[1] == 3:
            # get units for z axis from plotParams
            zu = plotParams['zu']
            
            if xu != '' and yu != '' and zu != '':
                title = '%s [$\\bf{z}$ @ (%0.2f %s, %0.2f %s, %0.2f %s)]' %(impulse_title,
                                                                            sourcePoints[index, 0], xu,
                                                                            sourcePoints[index, 1], yu,
                                                                            sourcePoints[index, 2], zu)
            elif xu == '' and yu == '' and zu == '':
                title = '%s [$\\bf{z}$ @ (%0.2f, %0.2f, %0.2f)]' %(impulse_title,
                                                                   sourcePoints[index, 0],
                                                                   sourcePoints[index, 1],
                                                                   sourcePoints[index, 2])
    
    return title

#==============================================================================
# Specific functions for plotting singular vectors...
                
def vector_title(flag, n, cmplx_part=None):
    '''
    Creates a plot title for each left/right singular vector
    Returns a raw formatted string using LaTeX
    
    Parameters:
    flag: a string, either 'left' or 'right'
    n: a nonnegative integer
    dtype: data type (real or complex)
    '''
    
    if flag == 'left':
        if cmplx_part == 'real':
            title = r'$\mathrm{Re}\{\widehat{\phi}_{%d}(\mathbf{x}_r,\nu)\}$' %(n)
        elif cmplx_part == 'imag':
            title = r'$\mathrm{Im}\{\widehat{\phi}_{%d}(\mathbf{x}_r,\nu)\}$' %(n)
        else:
            title = r'Left-Singular Vector $\phi_{%d}(\mathbf{x}_r,t)$' %(n)
            
    
    elif flag == 'right':
        if cmplx_part == 'real':
            title = r'$\mathrm{Re}\{\widehat{\psi}_{%d}(\mathbf{x}_s,\nu)\}$' %(n)
        elif cmplx_part == 'imag':
            title = r'$\mathrm{Im}\{\widehat{\psi}_{%d}(\mathbf{x}_s,\nu)\}$' %(n)
        else:
            title = r'Right-Singular Vector $\psi_{%d}(\mathbf{x}_s,t)$' %(n)
    
    return title
          


def process_key_vectors(event, xvals, rinterval, sinterval,
                        receiverPoints, sourcePoints, plotParams,
                        dtype=None):
    '''
    Determines how to draw the next plot based on keyboard events
    
    event: a keyboard hit, either 'left', 'right', 'up', or 'down' arrow keys
    
    Passed parameters:
    time: an array of time values over which the singular vectors are defined
    t0: left endpoint of the time axis
    tf: right endpoint of the time axis
    rinterval: an interval or sampling of the receivers used
    sinterval: an interval or sampling of the sources used
    receiverPoints: coordinates of the receivers
    sourcePoints: coordinates of the sources
    '''
    
    fig = event.canvas.figure
    ax1 = fig.axes[0]
    ax2 = fig.axes[1]
    
    if event.key == 'left' or event.key == 'down':
        if dtype == 'cmplx_left':
            fig.suptitle('Left-Singular Vector', color=ax1.titlecolor, fontsize=16)
            previous_vector(ax1, xvals, rinterval, receiverPoints, 'left', 'real', plotParams)
            previous_vector(ax2, xvals, rinterval, receiverPoints, 'left', 'imag', plotParams)
        elif dtype == 'cmplx_right':
            fig.suptitle('Right-Singular Vector', color=ax1.titlecolor, fontsize=16)
            previous_vector(ax1, xvals, sinterval, sourcePoints, 'right', 'real', plotParams)
            previous_vector(ax2, xvals, sinterval, sourcePoints, 'right', 'imag', plotParams)
        else:
            previous_vector(ax1, xvals, rinterval, receiverPoints, 'left', None, plotParams)
            previous_vector(ax2, xvals, sinterval, sourcePoints, 'right', None, plotParams)
    
    elif event.key == 'right' or event.key == 'up':
        if dtype == 'cmplx_left':
            fig.suptitle('Left-Singular Vector', color=ax1.titlecolor, fontsize=16)
            next_vector(ax1, xvals, rinterval, receiverPoints, 'left', 'real', plotParams)
            next_vector(ax2, xvals, rinterval, receiverPoints, 'left', 'imag', plotParams)
        elif dtype == 'cmplx_right':
            fig.suptitle('Right-Singular Vector', color=ax1.titlecolor, fontsize=16)
            next_vector(ax1, xvals, sinterval, sourcePoints, 'right', 'real', plotParams)
            next_vector(ax2, xvals, sinterval, sourcePoints, 'right', 'imag', plotParams)
        else:
            next_vector(ax1, xvals, rinterval, receiverPoints, 'left', None, plotParams)
            next_vector(ax2, xvals, sinterval, sourcePoints, 'right', None, plotParams)
    
    fig.canvas.draw()
            


def next_vector(ax, xvals, interval, coordinates, flag, cmplx_part, plotParams):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[2]
    title = vector_title(flag, ax.index + 1, cmplx_part)
    if cmplx_part is None:
        plotWiggles(ax, volume[:, :, ax.index], xvals, interval, 
                    coordinates, title, flag, plotParams)
    else:
        plotFreqVectors(ax, volume[:, :, ax.index], xvals, interval, 
                    coordinates, title, flag, plotParams)



def previous_vector(ax, xvals, interval, coordinates, flag, cmplx_part, plotParams):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[2]  # wrap around using %
    title = vector_title(flag, ax.index + 1, cmplx_part)
    if cmplx_part is None:
        plotWiggles(ax, volume[:, :, ax.index], xvals, interval, 
                    coordinates, title, flag, plotParams)
    else:
        plotFreqVectors(ax, volume[:, :, ax.index], xvals, interval, 
                    coordinates, title, flag, plotParams)
    
    

#==============================================================================
# Specific functions for plotting images...    
def process_key_images(event, plotParams, alpha, X, Y, Z, Ntau, tau):
    fig = event.canvas.figure
    ax = fig.axes[0]
    
    if event.key == 'left' or event.key == 'down':
        previous_image(ax, plotParams, alpha, X, Y, Z, Ntau, tau)
    
    elif event.key == 'right' or event.key == 'up':
        next_image(ax, plotParams, alpha, X, Y, Z, Ntau, tau)
    
    fig.canvas.draw()

def previous_image(ax, plotParams, alpha, X, Y, Z, Ntau, tau):
    volume = ax.volume
    ax.index = (ax.index - 1) % Ntau  # wrap around using %
    if Z is None:
        image_viewer(ax, volume[:, :, ax.index], plotParams,
                     alpha, X, Y, Z, tau[ax.index])
    else:
        image_viewer(ax, volume[:, :, :, ax.index], plotParams,
                     alpha, X, Y, Z, tau[ax.index])
    
def next_image(ax, plotParams, alpha, X, Y, Z, Ntau, tau):
    volume = ax.volume
    ax.index = (ax.index + 1) % Ntau  # wrap around using %
    if Z is None:
        image_viewer(ax, volume[:, :, ax.index], plotParams,
                     alpha, X, Y, Z, tau[ax.index])
    else:
        image_viewer(ax, volume[:, :, :, ax.index], plotParams,
                     alpha, X, Y, Z, tau[ax.index])
