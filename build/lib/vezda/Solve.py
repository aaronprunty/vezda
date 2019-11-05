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
from scipy.linalg import norm
from vezda.plot_utils import FontColor
from vezda.data_utils import load_data, load_impulse_responses
from vezda.LinearSamplingClass import LinearSamplingProblem

def info():
    commandName = FontColor.BOLD + 'vzsolve:' + FontColor.END
    description = ' solve for the unknown source function to obtain an image'
    
    return commandName + description

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nfe', action='store_true',
                        help='Solve the near-field equation (NFE).')
    parser.add_argument('--lse', action='store_true',
                        help='Solve the Lippmann-Schwinger equation (LSE).')
    parser.add_argument('--ngs', '-n', action='store_true',
                        help='''Normalize the impulse response by its energy. (Only used when
                        solving the near-field equation (NFE).''')
    parser.add_argument('--domain', '-d', type=str, default='freq', choices=['time', 'freq'],
                        help='''Specify whether to solve the linear system in the time domain
                        or frequency domain. Default is set to frequency domain for faster
                        performance.''')
    parser.add_argument('--method', '-m', type=str, default='lsmr', choices=['lsmr', 'lsqr', 'svd'],
                        help='''Specify the method for solving the linear system of equations:
                        iterative least-squares (lsmr/lsqr) or singular-value decomposition (svd).''')
    parser.add_argument('--nproc', type=int,
                        help='''Specify the number of processors to parallelize over. Default is serial
                        (i.e., one processor). nproc=-1 uses all available processors. nproc=-2 uses all
                        but one available processors.''')
    parser.add_argument('--numVals', '-k', type=int,
                        help='''Specify the number of singular values/vectors to compute.
                        Must a positive integer between 1 and the order of the linear operator.''')
    parser.add_argument('--regPar', '--alpha', type=float,
                        help='''Specify the value of the regularization parameter. Default is set
                        to zero.''')
    parser.add_argument('--atol', type=float,
                        help='''Specify the error tolerance of the linear operator A. (1e-q roughly
                        corresponds to 'q' correct decimal digits. Default is set to 1e-8.''')
    parser.add_argument('--btol', type=float,
                        help='''Specify the error tolerance of the right-hand side vectors b. (1e-q roughly
                        corresponds to 'q' correct decimal digits. Default is set to 1e-8.''')
    parser.add_argument('--medium', type=str, default='constant', choices=['constant', 'variable'],
                        help='''Specify whether the background medium is constant or variable
                        (inhomogeneous). If argument is set to 'constant', the velocity defined in
                        the required 'pulsesFun.py' file is used. Default is set to 'constant'.''')
    args = parser.parse_args()
    
    #==========================================================================
    # Check the value of the regularization parameter
    #==========================================================================
    if args.regPar is not None:
        if args.regPar >= 0.0:
            alpha = args.regPar
        else:
            sys.exit(textwrap.dedent(
                    '''
                    Error: Optional argument '--regPar/--alpha' cannot be negative. 
                    The regularization parameter must be greater than or equal to zero.
                    '''))
    else:
        # if args.regPar is None
        alpha = 0.0
        
    #==========================================================================
    # Check the value of the error tolerance of the linear operator
    #==========================================================================
    if args.atol is not None:
        if args.atol > 0.0:
            atol = args.atol
        else:
            sys.exit(textwrap.dedent(
                    '''
                    Error: Optional argument '--atol' must be positive. 
                    '''))
    else:
        # if args.atol is None
        atol = 1.0e-8
        
    #==========================================================================
    # Check the value of the tolerance for convergence of iterative methods
    #==========================================================================
    if args.btol is not None:
        if args.btol > 0.0:
            btol = args.btol
        else:
            sys.exit(textwrap.dedent(
                    '''
                    Error: Optional argument '--btol' must be positive. 
                    '''))
    else:
        # if args.btol is None
        btol = 1.0e-8
        
    #==========================================================================
    # Check the number of processors specified
    #==========================================================================
    if args.nproc is not None:
        if args.nproc != 0:
            nproc = args.nproc
        else:
            sys.exit(textwrap.dedent(
                    '''
                    Error: Optional argument '--nproc' must be nonzero. 
                    '''))
    else:
        # if args.nproc is None
        nproc = 1
        
    #==========================================================================
    # load data, impulseResponses
    # determine whether to solve near-field equation or Lippmann-Schwinger equation
    #==========================================================================
    data = load_data(args.domain, taper=True, verbose=True)
    impulseResponses = load_impulse_responses(args.domain, args.medium)
        
    if args.nfe:
        # Solve using the linear sampling method
        if args.ngs:
            print('Normalizing impulse responses by their energy...')
            for k in range(impulseResponses.shape[2]):
                impulseResponses[:, :, k] /= norm(impulseResponses[:, :, k])
        p = LinearSamplingProblem(operatorName='nfo', kernel=data, rhs_vectors=impulseResponses)
    
    elif args.lse:
        # Solve using the mimetic sampling method
        if args.domain == 'time':
            # This is particular to solving the Lippmann-Schwinger equation in the time domain
            # Pad data in the time domain to length 2*Nt-1 (length of circular convolution)
            N = data.shape[1] - 1
            npad = ((0, 0), (N, 0), (0, 0))
            data = np.pad(data, pad_width=npad, mode='constant', constant_values=0)
        
        p = LinearSamplingProblem(operatorName='lso', kernel=impulseResponses, rhs_vectors=data)
        
    else:
        userResponded = False
        print(textwrap.dedent(
              '''
              Please specify which linear equation you would like to solve:
                  
              Enter 'nfe' to solve the near-field equation. (Default)
              Enter 'lse' to solve the Lippmann-Schwinger equation.
              Enter 'q/quit' to exit.
              '''))
        while userResponded == False:
            answer = input('Action: ')
            if answer == '' or answer == 'nfe':
                args.nfe = True
                print('Solving the near-field equation...')
                if args.ngs:
                    print('Normalizing impulse responses by their energy...')
                    for k in range(impulseResponses.shape[2]):
                        impulseResponses[:, :, k] /= norm(impulseResponses[:, :, k])
                p = LinearSamplingProblem(operatorName='nfo', kernel=data, rhs_vectors=impulseResponses)
                userResponded = True
                break
            
            elif answer == 'lse':
                args.lse = True
                print('Solving the Lippmann-Schwinger equation...')
                if args.domain == 'time':
                    # This is particular to solving the Lippmann-Schwinger equation in the time domain
                    # Pad data in the time domain to length 2*Nt-1 (length of circular convolution)
                    N = data.shape[1] - 1
                    npad = ((0, 0), (N, 0), (0, 0))
                    data = np.pad(data, pad_width=npad, mode='constant', constant_values=0)
                
                p = LinearSamplingProblem(operatorName='lso', kernel=impulseResponses, rhs_vectors=data)
                userResponded = True
                break
            
            elif answer == 'q' or answer == 'quit':
                sys.exit('Exiting program.\n')
            
            else:
                print('Invalid response. Please enter \'nfe\', \'lse\', or \'q/quit\'.')
        
    #==========================================================================
    # Solve the problem using the specified method, regularization parameter,
    # and tolerance.
    # Construct images from solutions.
    # Save solutions and images to file.
    #==========================================================================
    # define extension for saving files
    if args.nfe:
        extension = 'NFE.npz'
    elif args.lse:
        extension = 'LSE.npz'
        
    X = p.solve(args.method, nproc, alpha, atol, btol, args.numVals)
    Image = p.construct_image(X)
        
    np.savez('solution'+extension, X=X, alpha=alpha, domain=args.domain)
    np.savez('image'+extension, Image=Image, method=args.method,
             alpha=alpha, atol=atol, btol=btol, domain=args.domain)