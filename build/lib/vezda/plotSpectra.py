# Copyright 2017-2019 Aaron C. Prunty
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
import sys
import argparse
import textwrap
import numpy as np
from vezda.plot_utils import setFigure, default_params, gradient_fill, zfunc
from vezda.data_utils import load_data, load_impulse_responses, get_user_windows
from vezda.signal_utils import compute_spectra
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import pickle
from vezda.plot_utils import FontColor

def info():
    commandName = FontColor.BOLD + 'vzspectra:' + FontColor.END
    description = ' plot the amplitude or power spectrum of time signals in the frequency domain'
    
    return commandName + description

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action='store_true',
                        help='Plot the frequency spectra of the recorded data. (Default)')
    parser.add_argument('--impulse', action='store_true',
                        help='Plot the frequency spectra of the simulated impulse responses.')
    parser.add_argument('--scaling', '-s', type=str, default='amp', choices=['amp', 'pow', 'psd'],
                        help='''Specify the scaling of the spectrum. Choose from amplitude ('amp'), power ('pow'),
                        or power spectral density ('psd') using Welch's method. Default is set to 'amp'.''')
    parser.add_argument('--nseg', type=int,
                        help='''Specify the approximate number of segments into which the time signal will be partitioned.
                        Only used if scaling is set to 'psd'. Increasing the number of segments increases computational
                        cost and the accuracy of the PSD estimate, but decreases frequency resolution. Default is set to 20.''')
    parser.add_argument('--fmin', type=float,
                        help='Specify the minimum frequency of the amplitude/power spectrum plot. Default is set to 0.')
    parser.add_argument('--fmax', type=float,
                        help='''Specify the maximum frequency of the amplitude/power spectrum plot. Default is set to the
                        maximum frequency bin based on the length of the time signal.''')
    parser.add_argument('--au', type=str,
                        help='Specify the amplitude units (e.g., Pa)')    
    parser.add_argument('--fu', type=str,
                        help='Specify the frequency units (e.g., Hz)')
    parser.add_argument('--format', '-f', type=str, default='pdf', choices=['png', 'pdf', 'ps', 'eps', 'svg'],
                        help='''specify the image format of the saved file. Accepted formats are png, pdf,
                        ps, eps, and svg. Default format is set to pdf.''')
    parser.add_argument('--mode', type=str, choices=['light', 'dark'], required=False,
                        help='''Specify whether to view plots in light mode for daytime viewing
                        or dark mode for nighttime viewing.
                        Mode must be either \'light\' or \'dark\'.''')
    args = parser.parse_args()
    
    #==============================================================================        
    # Get time window parameters
    tinterval, tstep, dt = get_user_windows()[1:4]
    datadir = np.load('datadir.npz')
    recordingTimes = np.load(str(datadir['recordingTimes']))
    recordingTimes = recordingTimes[tinterval]
    
    # Used for getting time and frequency units
    if Path('plotParams.pkl').exists():
        plotParams = pickle.load(open('plotParams.pkl', 'rb'))
    else:
        plotParams = default_params()
    
    if all(v is True for v in [args.data, args.impulse]):
        X = load_data(domain='time', verbose=True)
        Xlabel = plotParams['data_title']
        Xcolor = 'm'
        
        if 'testFuncs' not in datadir and not Path('VZImpulseResponses.npz').exists():
            X2 = load_impulse_responses(domain='time', medium='constant', verbose=True)
            X2label = plotParams['impulse_title']
            X2color = 'c'
        
        elif 'testFuncs' in datadir and not Path('VZImpulseResponses.npz').exists():
            X2 = load_impulse_responses(domain='time', medium='variable', verbose=True)
            X2label = plotParams['impulse_title']
            X2color = 'c'
            
        elif not 'testFuncs' in datadir and Path('VZImpulseResponses.npz').exists():
            X2 = load_impulse_responses(domain='time', medium='constant', verbose=True)
            X2label = plotParams['impulse_title']
            X2color = 'c'
            
        elif 'testFuncs' in datadir and Path('VZImpulseResponses.npz').exists():
            userResponded = False
            print(textwrap.dedent(
                 '''
                 Two files are available containing simulated impulse responses.
                 
                 Enter '1' to view the user-provided impulse responses. (Default)
                 Enter '2' to view the impulse responses computed by Vezda.
                 Enter 'q/quit' to exit.
                 '''))
            while userResponded == False:
                answer = input('Action: ')
                
                if answer == '' or answer == '1':
                    X2 = load_impulse_responses(domain='time', medium='variable', verbose=True)
                    X2label = plotParams['impulse_title']
                    X2color = 'c'
                    userResponded = True
                    break
                
                elif answer == '2':
                    X2 = load_impulse_responses(domain='time', medium='constant', verbose=True)
                    X2label = plotParams['impulse_title']
                    X2color = 'c'
                    userResponded = True
                    break
                
                elif answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.')
                
                else:
                    print('Invalid response. Please enter \'1\', \'2\', or \'q/quit\'.')
    
    elif (args.data and not args.impulse) or all(v is not True for v in [args.data, args.impulse]):
        # default is to plot spectra of data if user does not specify either args.data or args.impulse
        X = load_data(domain='time', verbose=True)
        Xlabel = plotParams['data_title']
        Xcolor = 'm'
        X2 = None
        
    elif not args.data and args.impulse:
        if 'testFuncs' not in datadir and not Path('VZImpulseResponses.npz').exists():
            X = load_impulse_responses(domain='time', medium='constant', verbose=True)
            Xlabel = plotParams['impulse_title']
            Xcolor = 'c'
        
        elif 'testFuncs' in datadir and not Path('VZImpulseResponses.npz').exists():
            X = load_impulse_responses(domain='time', medium='variable', verbose=True)
            Xlabel = plotParams['impulse_title']
            Xcolor = 'c'
            
        elif not 'testFuncs' in datadir and Path('VZImpulseResponses.npz').exists():
            X = load_impulse_responses(domain='time', medium='constant', verbose=True)
            Xlabel = plotParams['impulse_title']
            Xcolor = 'c'
                    
        elif 'testFuncs' in datadir and Path('VZImpulseResponses.npz').exists():
            userResponded = False
            print(textwrap.dedent(
                 '''
                 Two files are available containing simulated impulse responses.
                 
                 Enter '1' to view the user-provided impulse responses. (Default)
                 Enter '2' to view the impulse responses computed by Vezda.
                 Enter 'q/quit' to exit.
                 '''))
            while userResponded == False:
                answer = input('Action: ')
                
                if answer == '' or answer == '1':
                    X = load_impulse_responses(domain='time', medium='variable', verbose=True)
                    Xlabel = plotParams['impulse_title']
                    Xcolor = 'c'
                    userResponded = True
                    break
                
                elif answer == '2':
                    X = load_impulse_responses(domain='time', medium='constant', verbose=True)
                    Xlabel = plotParams['impulse_title']
                    Xcolor = 'c'
                    userResponded = True
                    break
                
                elif answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.')
                
                else:
                    print('Invalid response. Please enter \'1\', \'2\', or \'q/quit\'.')
        
        X2 = None
        
    #==============================================================================
    # compute spectra
    if args.nseg is not None:
        if args.nseg >= 1:
            nseg = args.nseg
        else:
            sys.exit(textwrap.dedent(
                    '''
                    Error: Optional argument '--nseg' must be greater than or equal to one.
                    '''))
    else:
        # if args.nseg is None
        nseg = 1
    
    freqs, A = compute_spectra(X, tstep * dt, scaling=args.scaling, nseg=nseg)
    if X2 is not None:
        Nt = X.shape[1]
        X2 = X2[:, -Nt:, :]
        freqs2, A2 = compute_spectra(X2, tstep * dt, scaling=args.scaling, nseg=nseg)
    else:
        freqs2 = None
        A2 = None
        
    if args.au is not None:
        plotParams['au'] = args.au
    
    if args.fu is not None:
        plotParams['fu'] = args.fu

    au = plotParams['au']
    fu = plotParams['fu']
    
    if args.scaling == 'amp':
        plotLabel = 'amplitude'
        plotParams['freq_title'] = 'Mean Amplitude Spectrum'
        if au != '':
            plotParams['freq_ylabel'] = 'Amplitude (%s)' %(au)
        else:
            plotParams['freq_ylabel'] = 'Amplitude'
   
    elif args.scaling == 'pow':
        plotLabel = 'power'
        plotParams['freq_title'] = 'Mean Power Spectrum'
        if au != '':
            plotParams['freq_ylabel'] = 'Power (%s)' %(au + '$^2$')
        else:
            plotParams['freq_ylabel'] = 'Power'
    
    elif args.scaling == 'psd':
        plotLabel = 'psd'
        plotParams['freq_title'] = 'Mean Power Spectral Density'
        if au != '' and fu != '':
            plotParams['freq_ylabel'] = 'Power/Frequency (%s)' %(au + '$^2/$' + fu)
        else:
            plotParams['freq_ylabel'] = 'Power/Frequency'
        
    if args.fmin is not None: 
        if args.fmin >= 0:
            if args.fmax is not None:
                if args.fmax > args.fmin:
                    plotParams['fmin'] = args.fmin
                    plotParams['fmax'] = args.fmax
                else:
                    sys.exit(textwrap.dedent(
                            '''
                            RelationError: The maximum frequency of the %s spectrum plot must
                            be greater than the mininum frequency.
                            ''' %(plotLabel)))   
            else:
                fmax = plotParams['fmax']
                if fmax > args.fmin:
                    plotParams['fmin'] = args.fmin
                else:
                    sys.exit(textwrap.dedent(
                            '''
                            RelationError: The specified minimum frequency of the %s spectrum 
                            plot must be less than the maximum frequency.
                            ''' %(plotLabel)))                                        
        else:
            sys.exit(textwrap.dedent(
                    '''
                    ValueError: The specified minimum frequency of the %s spectrum 
                    plot must be nonnegative.
                    ''' %(plotLabel)))
            
    #===============================================================================
    if args.fmax is not None:
        if args.fmin is not None:
            if args.fmin >= 0:
                if args.fmax > args.fmin:
                    plotParams['fmin'] = args.fmin
                    plotParams['fmax'] = args.fmax
                else:
                    sys.exit(textwrap.dedent(
                            '''
                            RelationError: The maximum frequency of the %s spectrum plot must
                            be greater than the mininum frequency.
                            ''' %(plotLabel)))
            else:
                sys.exit(textwrap.dedent(
                        '''
                        ValueError: The specified minimum frequency of the %s spectrum 
                        plot must be nonnegative.
                        ''' %(plotLabel)))
        else:
            fmin = plotParams['fmin']
            if args.fmax > fmin:
                plotParams['fmax'] = args.fmax
            else:
                sys.exit(textwrap.dedent(
                        '''
                        RelationError: The specified maximum frequency of the %s spectrum 
                        plot must be greater than the minimum frequency.
                        ''' %(plotLabel)))
    elif plotParams['fmax'] is None:
        plotParams['fmax'] = np.max(freqs)
                
    #===================================================================================
    if args.mode is not None:
        plotParams['view_mode'] = args.mode
    
    pickle.dump(plotParams, open('plotParams.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    
    fig, ax = setFigure(num_axes=1, mode=plotParams['view_mode'])
    
    if args.scaling == 'psd':
        plotscale = 'log'
    else:
        plotscale = 'linear'
    
    gradient_fill(freqs, A, fill_color=Xcolor, ax=ax, scale=plotscale, zorder=2)
    handles, labels = [], []
    handles.append(Line2D([0], [0], color=Xcolor, lw=4))
    labels.append(Xlabel)
    if all(v is not None for v in [freqs2, A2]):
        gradient_fill(freqs2, A2, fill_color=X2color, ax=ax, scale=plotscale, zorder=1)
        handles.append(Line2D([0], [0], color=X2color, lw=4))
        labels.append(X2label)
        
    ax.legend(handles, labels, fancybox=True, framealpha=1, shadow=True, loc='upper right')
    ax.set_title(plotParams['freq_title'], color=ax.titlecolor)
    
    fmin = plotParams['fmin']
    fmax = plotParams['fmax']
    if fu != '':
        ax.set_xlabel('Frequency (%s)' %(fu), color=ax.labelcolor)
    else:
        ax.set_xlabel('Frequency', color=ax.labelcolor)
    ax.set_ylabel(plotParams['freq_ylabel'], color=ax.labelcolor)
    ax.set_xlim([fmin, fmax])
    if args.scaling != 'psd':
        ax.set_ylim(bottom=0)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    fig.savefig(plotLabel + 'Spectrum.' + args.format, format=args.format, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()
    