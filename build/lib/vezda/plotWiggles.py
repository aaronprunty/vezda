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
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from vezda.data_utils import get_user_windows, load_data, load_impulse_responses, get_unique_indices
from vezda.plot_utils import default_params, Experiment, Plotter
from vezda.plot_utils import FontColor

def info():
    commandName = FontColor.BOLD + 'vzwiggles:' + FontColor.END
    description = ' plot time signals as waveforms'
    
    return commandName + description

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action='store_true',
                        help='Plot the recorded data. (Default)')
    parser.add_argument('--impulse', action='store_true',
                        help='Plot the simulated impulse responses.')
    parser.add_argument('--medium', type=str, default='constant', choices=['constant', 'variable'],
                        help='''Specify whether the background medium is constant or variable
                        (inhomogeneous). If argument is set to 'constant', the velocity defined in
                        the required 'pulsesFun.py' file is used. Default is set to 'constant'.''')
    parser.add_argument('--tu', type=str,
                        help='Specify the time units (e.g., \'s\' or \'ms\').')
    parser.add_argument('--au', type=str,
                        help='Specify the amplitude units (e.g., \'m\' or \'mm\').')
    parser.add_argument('--pclip', type=float,
                        help='''Specify the percentage (0-1) of the peak amplitude to display. This
                        parameter is used for pcolormesh plots only. Default is set to 1.''')
    parser.add_argument('--title', type=str,
                        help='''Specify a title for the wiggle plot. Default title is
                        \'Data\' if \'--data\' is passed and 'Impulse Response' if \'--impulse\'
                        is passed.''')
    parser.add_argument('--format', '-f', type=str, default='pdf', choices=['png', 'pdf', 'ps', 'eps', 'svg'],
                        help='''Specify the image format of the saved file. Accepted formats are png, pdf,
                        ps, eps, and svg. Default format is set to pdf.''')
    parser.add_argument('--map', action='store_true',
                        help='''Plot a map of the receiver and source/search point locations. The current
                        source/search point will be highlighted. The boundary of the scatterer will also
                        be shown if available.''')
    parser.add_argument('--mode', type=str, choices=['light', 'dark'], required=False,
                        help='''Specify whether to view plots in light mode for daytime viewing
                        or dark mode for nighttime viewing.
                        Mode must be either \'light\' or \'dark\'.''')
    
    args = parser.parse_args()
    #==============================================================================
    # if a plotParams.pkl file already exists, load relevant parameters
    if Path('plotParams.pkl').exists():
        plotParams = pickle.load(open('plotParams.pkl', 'rb'))
        
        # update parameters for wiggle plots based on passed arguments
        if args.mode is not None:
            plotParams['view_mode'] = args.mode
        
        if args.tu is not None:
            plotParams['tu'] = args.tu
        
        if args.au is not None:
            plotParams['au'] = args.au
            
        if args.pclip is not None:
            if args.pclip >= 0 and args.pclip <= 1:
                plotParams['pclip'] = args.pclip
            else:
                print(textwrap.dedent(
                      '''
                      Warning: Invalid value passed to argument \'--pclip\'. Value must be
                      between 0 and 1.
                      '''))
            
        if args.title is not None:
            if args.data:
                plotParams['data_title'] = args.title
            elif args.impulse:
                plotParams['tf_title'] = args.title
    
    else: # create a plotParams dictionary file with default values
        plotParams = default_params()
        
        # update parameters for wiggle plots based on passed arguments
        if args.mode is not None:
            plotParams['view_mode'] = args.mode
        
        if args.tu is not None:
            plotParams['tu'] = args.tu
        
        if args.au is not None:
            plotParams['au'] = args.au
        
        if args.title is not None:
            if args.data:
                plotParams['data_title'] = args.title
            elif args.impulse:
                plotParams['tf_title'] = args.title
    
    pickle.dump(plotParams, open('plotParams.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

    #==============================================================================
    # Load the relevant data to plot
    datadir = np.load('datadir.npz')
    receiverPoints = np.load(str(datadir['receivers']))
    time = np.load(str(datadir['recordingTimes']))
    
    # Apply any user-specified windows
    receiverNumbers, tinterval, tstep, dt, sourceNumbers = get_user_windows()
    receiverPoints = receiverPoints[receiverNumbers, :]
    time = time[tinterval]
    
    if 'sources' in datadir:
        sourcePoints = np.load(str(datadir['sources']))
        sourcePoints = sourcePoints[sourceNumbers, :]
            
        # Check for source-receiver reciprocity
        reciprocalNumbers = get_unique_indices(sourcePoints, receiverPoints)
        reciprocalNumbers = np.asarray(reciprocalNumbers, dtype=np.int)
            
        if len(reciprocalNumbers) > 0:
            newReceivers = sourcePoints[reciprocalNumbers, :]
            reciprocity = True
        else:
            reciprocity = False
    else:
        reciprocity = False
        sourcePoints = None
    
    # Load the scatterer boundary, if it exists
    if 'scatterer' in datadir:
        scatterer = np.load(str(datadir['scatterer']))
    else:
        scatterer = None    
    
    if all(v is True for v in [args.data, args.impulse]):
        # User specified both data and impulse response for plotting
        # Send error message and exit.
        sys.exit(textwrap.dedent(
                '''
                Error: Cannot plot both recorded data and simulated impulse responses. Use
                
                    vzwiggles --data
                    
                to plot the recorded data or
                
                    vzwiggles --impulse
                    
                to plot the simulated impulse responses.
                '''))
    
    elif all(v is not True for v in [args.data, args.impulse]) or args.data:
        # If user did not specify which wiggles to plot, plot recorded data by default.
        # load the 3D data array into variable 'X'
        # X[receiver, time, source]
        wiggleType = 'data'
        X = load_data(domain='time', verbose=True)
        
        if reciprocity:
            Nr = len(receiverNumbers)
            Ns = len(sourceNumbers)
            M = len(reciprocalNumbers)
            
            XR = X[-M:, :, -Nr:]
            X = X[:Nr, :, :Ns]
            
            reciprocalNumbers += 1        
            ER = Experiment(XR, time, reciprocalNumbers, newReceivers,
                            receiverNumbers, receiverPoints, wiggleType)                        
        else:
            ER = None
        
    elif args.impulse:
        wiggleType = 'impulse'
        
        # Update time to convolution times
        T = time[-1] - time[0]
        time = np.linspace(-T, T, 2 * len(time) - 1)
        X, sourcePoints = load_impulse_responses(domain='time', medium=args.medium,
                                                 verbose=True, return_search_points=True)
        
        # Update sourceNumbers to match search points
        sourceNumbers = np.arange(sourcePoints.shape[0])
        
        if reciprocity:
            Nr = len(receiverNumbers)
            M = len(reciprocalNumbers)
            
            XR = X[-M:, :, :]
            X = X[:Nr, :, :]
            
            reciprocalNumbers += 1        
            ER = Experiment(XR, time, reciprocalNumbers, newReceivers,
                            sourceNumbers, sourcePoints, wiggleType)
        else:
            ER = None
    
    #==============================================================================        
    # increment source/receiver numbers to be consistent with
    # one-based indexing (i.e., count from one instead of zero)
    sourceNumbers += 1
    receiverNumbers += 1
    
    E = Experiment(X, time, receiverNumbers, receiverPoints,
                   sourceNumbers, sourcePoints, wiggleType)
        
    p = Plotter(E, ER)
    p.plot(scatterer, plotParams, args.map)
    plt.show()
