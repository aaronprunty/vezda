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
import numpy as np
from scipy.linalg import norm
from tqdm import trange
from time import sleep
import subprocess

def free_space_ir(receiverPoints, recordingTimes, sourcePoint, velocity, pulseFunc):
    '''
    Computes the impulse response for two- or three-dimensional free space. An impulse response
    is simply the free-space Green function convolved with a smooth, time-dependent pulse
    function.
    
     Inputs:
         receiverPoints: Nr x (2,3) array, specifies the coordinates of the 
              receiver points, where 'Nr' is the number of the points.
         recordingTimes: recording time array (row or column vector of length 'Nt')
         sourcePoint: 1 x (2,3) array specifying the source point. 
         velocity: a scalar specifying the wave speed
         pulseFunc: a function handle, gives the time depedence of the pulse function
    
     Output:
         impulseResponse: Nr x Nt array containing the computed impulse response
    '''
    # get the number of space dimensions (2 or 3)
    dim = receiverPoints.shape[1]
    
    # compute the distance between the receiver points and the source point
    r = norm(receiverPoints - sourcePoint, axis=1) # ||x - z||
    
    T, R = np.meshgrid(recordingTimes, r)
    retardedTime = T - R / velocity
    
    # get machine precision
    eps = np.finfo(float).eps     # about 2e-16 (so we never divide by zero)
    pulse = pulseFunc(retardedTime)
    if dim == 2:
        sqrtTR = np.lib.scimath.sqrt(T**2 - (R / velocity)**2)
        impulseResponse = np.divide(pulse, 2 * np.pi * sqrtTR + eps)
    
    elif dim == 3:
        impulseResponse = np.divide(pulse, 4 * np.pi * R + eps)
        
        
    # derivative of the pulse function
    #tau = retardedTime # = recordingTimes
    #dpulse = (4*cos(4*tau)-3.2*(tau-3).*sin(4*tau)).*exp(-1.6*(tau-3).^2);
    #r = receiverPoints - samplingPoint # x - z
    #nuxz = np.dot(dipoleMoment, r)
    #NXXZ = repmat(nuxz,1,length(t));
    #rhsFun = NXXZ.*(R.*pulse-sTR.*dpulse)./(2*pi*sTR.^3); % dipole
        
    impulseResponse[retardedTime<=0] = 0    # causality     
    
    return impulseResponse.real


def call_to_other_func(receiverPoints, recordingTimes, sourcePoints):
    
    command = input('$ ')
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()  # receive output from the python2 script
    
    filepath = input('where is the file?: ')
    impulseResponse = np.load(filepath)
    
    return impulseResponse


def compute_impulse_responses(medium, receiverPoints, recordingTimes, searchPoints, velocity, pulse):
    '''
    Compute the impulse responses for a specified medium and search grid.
    
    Inputs:
    receiverPoints: an array of the receiver locations in 2D or 3D space
    recordingTimes: an array of the recording times
    searchPoints: an array of search points in 2D or 3D space
    velocity: the velocity of the (constant) medium through which the waves propagate
    pulse: a function of time that describes the shape of the wave
    '''
    
    # get the number of receivers, time samples, and sources
    Nr = receiverPoints.shape[0]
    Nt = len(recordingTimes)
    Ns = searchPoints.shape[0]
        
    if Nr < Ns:
        # Use source-receiver reciprocity to efficiently compute impulse responses
        
        if medium == 'constant':
            impulseResponses = np.zeros((Ns, Nt, Nr))
            for i in trange(Nr):
                impulseResponses[:, :, i] = free_space_ir(searchPoints, recordingTimes,
                                receiverPoints[i, :], velocity, pulse)
                sleep(0.001)
        
        else:
            impulseResponses = call_to_other_func(searchPoints, recordingTimes, receiverPoints)
        
        impulseResponses = np.swapaxes(impulseResponses, 0, 2)
        
    else:
        if medium == 'constant':
            impulseResponses = np.zeros((Nr, Nt, Ns))
            for i in trange(Ns):
                impulseResponses[:, :, i] = free_space_ir(receiverPoints, recordingTimes,
                          searchPoints[i, :], velocity, pulse)
                sleep(0.001)
        
        else:
            impulseResponses = call_to_other_func(receiverPoints, recordingTimes, searchPoints)
        
    return impulseResponses


def samplingIsCurrent(Dict, receiverPoints, recordingTimes, searchPoints, tau, velocity, peakFreq=None, peakTime=None):
    if peakFreq is None and peakTime is None:
        
        if np.array_equal(Dict['searchPoints'], searchPoints) and np.array_equal(Dict['tau'], tau):
            print('Search grid and focusing time are consistent...')
        
            if Dict['velocity'] == velocity:
                print('Background velocity is consistent...')
                    
                if np.array_equal(Dict['receivers'], receiverPoints):
                    print('Receiver points are consistent...')
                        
                    if np.array_equal(Dict['time'], recordingTimes):
                        print('Recording time interval is consistent...')
                        return True
                        
                    else:
                        print('Current recording time interval is inconsistent...')
                        return False
                    
                else:
                    print('Current receiver points are inconsistent...')
                    return False
                    
            else:
                print('Current pulse function is inconsistent...')
                return False
        
        else:
            print('Current search grid or focusing time is inconsistent...')
            return False
            
    else:
        
        if np.array_equal(Dict['searchPoints'], searchPoints) and np.array_equal(Dict['tau'], tau):
            print('Search grid and focusing time are consistent...')
            
            if Dict['peakFreq'] == peakFreq and Dict['peakTime'] == peakTime and Dict['velocity'] == velocity:
                print('Pulse function and background velocity are consistent...')
                    
                if np.array_equal(Dict['receivers'], receiverPoints):
                    print('Receiver points are consistent...')
                        
                    if np.array_equal(Dict['time'], recordingTimes):
                        print('Recording time interval is consistent...')
                        return True
                
                    else:
                        print('Current recording time interval is inconsistent...')
                        return False                                            
                    
                else:
                    print('Current receiver points are inconsistent...')
                    return False
                    
            else:
                print('Current pulse function is inconsistent...')
                return False
        
        else:
            print('Current search grid or focusing time is inconsistent...')
            return False