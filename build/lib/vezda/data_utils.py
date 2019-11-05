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
import os
import sys
import numpy as np
import pickle
from pathlib import Path
import textwrap
from vezda.math_utils import nextPow2
from vezda.signal_utils import tukey_taper
from vezda.sampling_utils import samplingIsCurrent, compute_impulse_responses
from vezda.plot_utils import default_params
sys.path.append(os.getcwd())
import pulseFun

datadir = np.load('datadir.npz')

# Used for getting time and frequency units
if Path('plotParams.pkl').exists():
    plotParams = pickle.load(open('plotParams.pkl', 'rb'))
else:
    plotParams = default_params()

def load_data(domain, taper=False, verbose=False):
    # load the recorded data    
    print('Loading recorded waveforms...')
    if Path('noisyData.npz').exists():
        userResponded = False
        print(textwrap.dedent(
              '''
              Detected that band-limited noise has been added to the data array.
              Would you like to use the noisy data? ([y]/n)
              
              Enter 'q/quit' exit the program.
              '''))
        while userResponded == False:
            answer = input('Action: ')
            if answer == '' or answer == 'y' or answer == 'yes':
                print('Proceeding with noisy data...')
                # read in the noisy data array
                data = np.load('noisyData.npz')['noisyData']
                userResponded = True
            elif answer == 'n' or answer == 'no':
                print('Proceeding with noise-free data...')
                # read in the recorded data array
                data = np.load(str(datadir['recordedData']))
                userResponded = True
            elif answer == 'q' or answer == 'quit':
                sys.exit('Exiting program.\n')
            else:
                print('Invalid response. Please enter \'y/yes\', \'n\no\', or \'q/quit\'.')
                
    else:
        # read in the recorded data array
        data = np.load(str(datadir['recordedData']))
        
    # apply user-specified windows to data array
    rinterval, tinterval, tstep, dt, sinterval = get_user_windows(verbose)
    print('Applying windows to data volume...')
    data = data[rinterval, :, :]
    data = data[:, tinterval, :]
    data = data[:, :, sinterval]
    
    # check if source-receiver reciprocity can be used
    if 'sources' in datadir:
        print('Source positions are known...')
        print('Checking if source-receiver reciprocity can be used...')
        receiverPoints = np.load(str(datadir['receivers']))
        receiverPoints = receiverPoints[rinterval, :]
            
        sourcePoints = np.load(str(datadir['sources']))
        sourcePoints = sourcePoints[sinterval, :]
            
        indices = get_unique_indices(sourcePoints, receiverPoints)
        
        N = len(indices)
        if N == 0:
            print('Sources and receivers are co-located. Reciprocity adds no value...')
        else:
            print('Adding reciprocal data for %d unique source points...' %(N))
            newData = data[:, :, indices]
            newData = np.swapaxes(newData, 0, 2)
            M = newData.shape[2]
            npad = ((0, N), (0, 0), (0, M))
            data = np.pad(data, pad_width=npad, mode='constant', constant_values=0)
            data[-N:, :, -M:] = newData
    
    if taper:
        # Apply tapered cosine (Tukey) window to time signals.
        # This ensures that any fast fourier transforms (FFTs) used
        # will be acting on a function that is continuous at its edges.
        data = tukey_taper(data, tstep * dt, pulseFun.peakFreq)
    
    if domain == 'freq':
        print('Transforming data to the frequency domain...')
        data = fft_and_window(data, tstep * dt, double_length=True)
        
    return data

def load_impulse_responses(domain, medium, verbose=False, return_search_points=False):
    # load user-specified windows
    rinterval, tinterval, tstep, dt = get_user_windows(verbose, skip_sources=True)
    
    receiverPoints = np.load(str(datadir['receivers']))
    recordingTimes = np.load(str(datadir['recordingTimes']))
        
    # Apply user-specified windows
    receiverPoints = receiverPoints[rinterval, :]
    recordingTimes = recordingTimes[tinterval]
        
    # load/compute the impulse responses and search points
    if 'impulseResponses' in datadir:
        impulseResponses = np.load(str(datadir['impulseResponses']))
        searchGrid = np.load(str(datadir['searchGrid']))
        loadVZImpulseResponses = False
    
    else:
        try:
            searchGrid = np.load('searchGrid.npz')
            loadVZImpulseResponses = True
        except FileNotFoundError:
            searchGrid = None
        
        if searchGrid is None:
            sys.exit(textwrap.dedent(
                    '''
                    A search grid needs to be set up before impulse responses can
                    be computed or loaded.
                    Enter:
                            
                        vzgrid --help
                
                    from the command-line for more information on how to set up a
                    search grid.
                    '''))        
            
    x = searchGrid['x']
    y = searchGrid['y']
    Nx, Ny = len(x), len(y)
    tau = searchGrid['tau']
    if 'z' in searchGrid:
        z = searchGrid['z']
        Nz = len(z)
        searchPoints = np.vstack(np.meshgrid(x, y, z, indexing='ij')).reshape(3, Nx * Ny * Nz).T
    else:
        searchPoints = np.vstack(np.meshgrid(x, y, indexing='ij')).reshape(2, Nx * Ny).T
            
    if loadVZImpulseResponses:
        # check if source-receiver reciprocity can be used
        if 'sources' in datadir:
            sinterval = get_user_windows()[-1]
            
            sourcePoints = np.load(str(datadir['sources']))
            sourcePoints = sourcePoints[sinterval, :]
            
            indices = get_unique_indices(sourcePoints, receiverPoints)
            if len(indices) > 0:
                receiverPoints = np.vstack((receiverPoints, sourcePoints[indices, :]))
        
        pulse = lambda t : pulseFun.pulse(t)
        velocity = pulseFun.velocity
        peakFreq = pulseFun.peakFreq
        peakTime = pulseFun.peakTime
            
        tu = plotParams['tu']
        # set up the convolution times based on length of recording time interval
        T = recordingTimes[-1] - recordingTimes[0]
        convolutionTimes = np.linspace(-T, T, 2 * len(recordingTimes) - 1)
        
        if Path('VZImpulseResponses.npz').exists():
            print('Detected that impulse responses have already been computed...')
            print('Checking consistency with current search grid, focusing time, and pulse function...')
            IRDict = np.load('VZImpulseResponses.npz')
                
            if samplingIsCurrent(IRDict, receiverPoints, convolutionTimes, searchPoints, tau, velocity, peakFreq, peakTime):
                impulseResponses = IRDict['IRarray']
                print('Impulse responses are up to date...')
                    
            else:
                if tau != 0.0:
                    if tu != '':
                        print('Recomputing impulse responses for current search grid and focusing time %0.2f %s...' %(tau, tu))
                    else:
                        print('Recomputing impulse responses for current search grid and focusing time %0.2f...' %(tau))
                    impulseResponses = compute_impulse_responses(medium, receiverPoints, convolutionTimes - tau,
                                                                 searchPoints, velocity, pulse)
                else:
                    print('Recomputing impulse responses for current search grid...')
                    impulseResponses = compute_impulse_responses(medium, receiverPoints, convolutionTimes,
                                                                 searchPoints, velocity, pulse)
                    
                np.savez('VZImpulseResponses.npz', IRarray=impulseResponses, time=convolutionTimes, receivers=receiverPoints,
                         peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                         searchPoints=searchPoints, tau=tau)
                    
        else:                
            if tau != 0.0:
                if tu != '':
                    print('Computing impulse responses for current search grid and focusing time %0.2f %s...' %(tau, tu))
                else:
                    print('Computing impulse responses for current search grid and focusing time %0.2f...' %(tau))
                impulseResponses = compute_impulse_responses(medium, receiverPoints, convolutionTimes - tau,
                                                             searchPoints, velocity, pulse)
            else:
                print('Computing impulse responses for current search grid...')
                impulseResponses = compute_impulse_responses(medium, receiverPoints, convolutionTimes,
                                                             searchPoints, velocity, pulse)
                    
            np.savez('VZImpulseResponses.npz', IRarray=impulseResponses, time=convolutionTimes, receivers=receiverPoints,
                     peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                     searchPoints=searchPoints, tau=tau)
        
    if domain == 'freq':
        print('Transforming impulse responses to the frequency domain...')
        impulseResponses = fft_and_window(impulseResponses, tstep * dt, double_length=False)
    
    if return_search_points:
        return impulseResponses, searchPoints
        
    else:
        return impulseResponses
        

def get_user_windows(verbose=False, skip_sources=False):
    recordingTimes = np.load(str(datadir['recordingTimes']))
    dt = recordingTimes[1] - recordingTimes[0]
    
    if Path('window.npz').exists():
        
        windowDict = np.load('window.npz')
        
        # Receiver window parameters
        rstart = windowDict['rstart']
        rstop = windowDict['rstop']
        rstep = windowDict['rstep']
        rinterval = np.arange(rstart, rstop, rstep)
        
        # Time window parameters (with units of time)
        tstart = windowDict['tstart']
        tstop = windowDict['tstop']
        tstep = windowDict['tstep']
        tu = plotParams['tu']
        
        if verbose:
            print('Detected user-specified windows:\n')
        
            # For display/printing purposes, count receivers with one-based
            # indexing. This amounts to incrementing the rstart parameter by 1
            print('window @ receivers : start =', rstart + 1)
            print('window @ receivers : stop =', rstop)
            print('window @ receivers : step =', rstep, '\n')
        
            if tu != '':
                print('window @ time : start = %0.2f %s' %(tstart, tu))
                print('window @ time : stop = %0.2f %s' %(tstop, tu))
            else:
                print('window @ time : start =', tstart)
                print('window @ time : stop =', tstop)
            print('window @ time : step =', tstep, '\n')
        
        # Convert time window parameters to corresponding array indices
        tstart = int(round(tstart / dt))
        tstop = int(round(tstop / dt))
        tinterval = np.arange(tstart, tstop, tstep)
        
        if skip_sources:
            return rinterval, tinterval, tstep, dt
        
        else:
            # Source window parameters
            slabel = windowDict['slabel']
            sstart = windowDict['sstart']
            sstop = windowDict['sstop']
            sstep = windowDict['sstep']
            sinterval = np.arange(sstart, sstop, sstep)
        
            if verbose:
                # For display/printing purposes, count recordings/sources with one-based
                # indexing. This amounts to incrementing the sstart parameter by 1
                print('window @ %s : start = %s' %(slabel, sstart + 1))
                print('window @ %s : stop = %s' %(slabel, sstop))
                print('window @ %s : step = %s\n' %(slabel, sstep))
            
            return rinterval, tinterval, tstep, dt, sinterval
        
    else:
        # Set default window parameters if user did
        # not specify window parameters.
        X = np.load(str(datadir['recordedData']))
        Nr, Nt, Ns = X.shape
        
        # Receiver window parameters
        rstart = 0
        rstop = Nr
        rstep = 1
        rinterval = np.arange(rstart, rstop, rstep)
        
        # Time window parameters (integers corresponding to array indices)
        tstart = 0
        tstop = Nt
        tstep = 1
        tinterval = np.arange(tstart, tstop, tstep)
        
        if skip_sources:
            return rinterval, tinterval, tstep, dt
            
        else:
            # Source window parameters
            sstart = 0
            sstop = Ns
            sstep = 1
            sinterval = np.arange(sstart, sstop, sstep)
            
            return rinterval, tinterval, tstep, dt, sinterval


#==============================================================================
def fft_and_window(X, dt, double_length):
    # Transform X into the frequency domain and apply window around nonzero
    # frequency components
    
    if double_length:
        N = nextPow2(2 * X.shape[1])
    else:
        N = nextPow2(X.shape[1])
    X = np.fft.rfft(X, n=N, axis=1)
    
    if plotParams['fmax'] is None:
        freqs = np.fft.rfftfreq(N, dt)
        plotParams['fmax'] = np.max(freqs)
        pickle.dump(plotParams, open('plotParams.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    
    # Apply the frequency window
    fmin = plotParams['fmin']
    fmax = plotParams['fmax']
    fu = plotParams['fu']   # frequency units (e.g., Hz)
    
    if fu != '':
        print('Applying frequency window: [%0.2f %s, %0.2f %s]' %(fmin, fu, fmax, fu))
    else:
        print('Applying frequency window: [%0.2f, %0.2f]' %(fmin, fmax))
        
    df = 1.0 / (N * dt)
    startIndex = int(round(fmin / df))
    stopIndex = int(round(fmax / df))
        
    finterval = np.arange(startIndex, stopIndex, 1)
    X = X[:, finterval, :]
    
    return X

#==============================================================================
def get_unique_indices(coordinates1, coordinates2):    
    N = coordinates1.shape[0]
    unique_indices = []
    for n in range(N):
        if (coordinates1[n, :] != coordinates2).any(axis=1).all():
            unique_indices.append(n)
        
    return unique_indices