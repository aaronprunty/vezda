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
import argparse
import textwrap
import numpy as np
from pathlib import Path
from vezda.plot_utils import FontColor

def info():
    commandName = FontColor.BOLD + 'vzdata:' + FontColor.END
    description = ' specify the directory containing the experimental data'
    
    return commandName + description

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None,
                        help='''specify the path to the directory containing the
                        experimental data.''')
    args = parser.parse_args()
    
    if args.path is None:
        if not Path('datadir.npz').exists():
            sys.exit(textwrap.dedent(
                    '''
                    A relative path to the data directory has not been specified for
                    the current working directory. To specify a relative path to the
                    data directory, enter:
                        
                        vzdata --path=<path/to/data/files>
                        
                    from the command line.
                    '''))
        
        elif Path('datadir.npz').exists():
            datadir = np.load('datadir.npz')
            dataPath = str(datadir['path'])
            files = list(datadir['files'])
            for f in range(len(files)):
                files[f] = files[f].split('/')[-1]
                
            print('\nCurrent data directory:\n')
            print(dataPath, '\n')
            print('Known files:\n')
            print(*files, sep='\n')
            sys.exit('')
    
    elif args.path is not None:
        dataPath = os.path.abspath(args.path)
        #==============================================================================
        if Path(os.path.join(dataPath, 'receiverPoints.npy')).exists():
            receivers = os.path.join(dataPath, 'receiverPoints.npy')
        else:
            userResponded = False
            print(textwrap.dedent(
                 '''
                 Error: Expected file \'receiverPoints.npy\' not found. Does a file
                 exist containing the receiver coordinates? (This is a required file.)
                 
                 Enter 'y/yes' to specify the filename containing the receiver coordinates
                 (must be binary NumPy '.npy' format). (Default)
                 Enter 'n/no' or 'q/quit to exit this program.
                 '''))
            while userResponded == False:
                answer = input('Action: ')
                if answer == '' or answer == 'y' or answer == 'yes':
                    receiverFile = input('Please specify the filename containing the receiver coordinates: ')
                    if '.npy' in receiverFile and Path(os.path.join(dataPath, receiverFile)).exists():
                        receivers = os.path.join(dataPath, receiverFile)
                        userResponded = True
                    elif '.npy' in receiverFile and not Path(os.path.join(dataPath, receiverFile)).exists():
                        print(textwrap.dedent(
                              '''
                              Error: File \'%s\' does not exist within the specified data directory.
                              
                              Enter 'y/yes' to specify another filename containing the receiver coordinates
                              (must be binary NumPy '.npy' format). (Default)
                              Enter 'n/no' or 'q/quit to exit this program.
                              ''' %(receiverFile)))
                    elif '.npy' not in receiverFile:
                        print(textwrap.dedent(
                              '''
                              Error: File \'%s\' is not NumPy '.npy' format.
                              
                              Enter 'y/yes' to specify another filename containing the receiver coordinates
                              (must be binary NumPy '.npy' format). (Default)
                              Enter 'n/no' or 'q/quit to exit this program.
                              ''' %(receiverFile)))
                elif answer == 'n' or answer == 'no' or answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.\n')                
                else:
                    print('Invalid response. Please enter \'y/yes\', \'n/no\', or \'q/quit\'.')
        
        #==============================================================================
        if Path(os.path.join(dataPath, 'sourcePoints.npy')).exists():
            sources = os.path.join(dataPath, 'sourcePoints.npy')
            noSources = False
        else:
            userResponded = False
            print(textwrap.dedent(
                 '''
                 Warning: Expected file \'sourcesPoints.npy\' not found. Does a file
                 exist containing the source coordinates? (This is NOT a required file.)
                 
                 Enter 'y/yes' to specify the filename containing the source coordinates
                 (must be binary NumPy '.npy' format).
                 Enter 'n/no' to proceed without specifying the source coordinates. (Default)
                 Enter 'q/quit to exit this program.
                 '''))
            while userResponded == False:
                answer = input('Action: ')
                if answer == 'y' or answer == 'yes':
                    sourceFile = input('Please specify the filename containing the source coordinates: ')
                    if '.npy' in sourceFile and Path(os.path.join(dataPath, sourceFile)).exists():
                        sources = os.path.join(dataPath, sourceFile)
                        noSources = False
                        userResponded = True
                    elif '.npy' in sourceFile and not Path(os.path.join(dataPath, sourceFile)).exists():
                        print(textwrap.dedent(
                              '''
                              Error: File \'%s\' does not exist within the specified data directory.
                                
                              Enter 'y/yes' to specify another filename containing the source coordinates
                              (must be binary NumPy '.npy' format).
                              Enter 'n/no' to proceed without specifying the source coordinates. (Default)
                              Enter 'q/quit to exit this program.
                              ''' %(sourceFile)))
                    elif '.npy' not in sourceFile:
                        print(textwrap.dedent(
                              '''
                              Error: File \'%s\' is not NumPy '.npy' format.
                              
                              Enter 'y/yes' to specify another filename containing the source coordinates
                              (must be binary NumPy '.npy' format).
                              Enter 'n/no' to proceed without specifying the source coordinates. (Default)
                              Enter 'q/quit to exit this program.
                              ''' %(sourceFile)))
                elif answer == '' or answer == 'n' or answer == 'no':
                    print('Proceeding without specifying the source coordinates.')
                    noSources = True
                    userResponded = True
                elif answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.\n')                
                else:
                    print('Invalid response. Please enter \'y/yes\', \'n/no\', or \'q/quit\'.')
    
        #==============================================================================
        if Path(os.path.join(dataPath, 'scattererPoints.npy')).exists():
            scatterer = os.path.join(dataPath, 'scattererPoints.npy')
            noScatterer = False
        else:
            userResponded = False
            print(textwrap.dedent(
                 '''
                 Warning: Expected file \'scattererPoints.npy\' not found. Does a file
                 exist containing the scatterer coordinates? (This is NOT a required file.)
                 
                 Enter 'y/yes' to specify the filename containing the scatterer coordinates
                 (must be binary NumPy '.npy' format).
                 Enter 'n/no' to proceed without specifying the scatterer coordinates. (Default)
                 Enter 'q/quit' to exit this program.
                 '''))
            while userResponded == False:
                answer = input('Action: ')
                if answer == 'y' or answer == 'yes':
                    scattererFile = input('Please specify the filename containing the scatterer coordinates: ')
                    if '.npy' in scattererFile and Path(os.path.join(dataPath, scattererFile)).exists():
                        scatterer = os.path.join(dataPath, scattererFile)
                        noScatterer = False
                        userResponded = True
                    elif '.npy' in scattererFile and not Path(os.path.join(dataPath, scattererFile)).exists():
                        print(textwrap.dedent(
                              '''
                              Error: File \'%s\' does not exist within the specified data directory.
                                
                              Enter 'y/yes' to specify another filename containing the scatterer coordinates
                              (must be binary NumPy '.npy' format).
                              Enter 'n/no' to proceed without specifying the scatterer coordinates. (Default)
                              Enter 'q/quit' to exit this program.
                              ''' %(scattererFile)))
                    elif '.npy' not in scattererFile:
                        print(textwrap.dedent(
                              '''
                              Error: File \'%s\' is not NumPy '.npy' format.
                              
                              Enter 'y/yes' to specify another filename containing the scatterer coordinates
                              (must be binary NumPy '.npy' format).
                              Enter 'n/no' to proceed without specifying the scatterer coordinates. (Default)
                              Enter 'q/quit' to exit this program.
                              ''' %(scattererFile)))
                elif answer == '' or answer == 'n' or answer == 'no':
                    print('Proceeding without specifying the scatterer coordinates.')
                    noScatterer = True
                    userResponded = True
                elif answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.\n')
                else:
                    print('Invalid response. Please enter \'y/yes\', \'n/no\', or \'q/quit\'.')
    
        #==============================================================================    
        if Path(os.path.join(dataPath, 'recordingTimes.npy')).exists():
            recordingTimes = os.path.join(dataPath, 'recordingTimes.npy')
        else:
            userResponded = False
            print(textwrap.dedent(
                 '''
                 Error: Expected file \'recordingTimes.npy\' not found. Does a file
                 exist containing the recording times? (This is a required file.)
                 
                 Enter 'y/yes' to specify the filename containing the recording
                 times (must be binary NumPy '.npy' format). (Default)
                 Enter 'n/no' or 'q/quit to exit this program.
                 '''))
            while userResponded == False:
                answer = input('Action: ')
                if answer == '' or answer == 'y' or answer == 'yes':
                    timeFile = input('Please specify the filename containing the recording times: ')
                    if '.npy' in timeFile and Path(os.path.join(dataPath, timeFile)).exists():
                        recordingTimes = os.path.join(dataPath, timeFile)
                        userResponded = True
                    elif '.npy' in timeFile and not Path(os.path.join(dataPath, timeFile)).exists():
                        print(textwrap.dedent(
                              '''
                              Error: File \'%s\' does not exist within the specified data directory.
                              
                              Enter 'y/yes' to specify another filename containing the recording times
                              (must be binary NumPy '.npy' format). (Default)
                              Enter 'n/no' or 'q/quit to exit this program.
                              ''' %(timeFile)))
                    elif '.npy' not in timeFile:
                        print(textwrap.dedent(
                              '''
                              Error: File \'%s\' is not NumPy '.npy' format.
                              
                              Enter 'y/yes' to specify another filename containing the recording times
                              (must be binary NumPy '.npy' format). (Default)
                              Enter 'n/no' or 'q/quit to exit this program.
                              ''' %(timeFile)))
                elif answer == 'n' or answer == 'no' or answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.\n')                
                else:
                    print('Invalid response. Please enter \'y/yes\', \'n/no\', or \'q/quit\'.')
    
        #==============================================================================    
        if Path(os.path.join(dataPath, 'recordedData.npy')).exists():
            recordedData = os.path.join(dataPath, 'recordedData.npy')
        else:
            userResponded = False
            print(textwrap.dedent(
                 '''
                 Error: Expected file \'recordedData.npy\' not found. Does a file
                 exist containing the recorded waves? (This is a required file.)
                 
                 Enter 'y/yes' to specify the filename containing the recorded
                 waves (must be binary NumPy '.npy' format). (Default)
                 Enter 'n/no' or 'q/quit to exit this program.
                 '''))
            while userResponded == False:
                answer = input('Action: ')
                if answer == '' or answer == 'y' or answer == 'yes':
                    dataFile = input('Please specify the filename containing the recorded waves: ')
                    if '.npy' in dataFile and Path(os.path.join(dataPath, dataFile)).exists():
                        recordedData = os.path.join(dataPath, dataFile)
                        userResponded = True
                    elif '.npy' in dataFile and not Path(os.path.join(dataPath, dataFile)).exists():
                        print(textwrap.dedent(
                              '''
                              Error: File \'%s\' does not exist within the specified data directory.
                              
                              Enter 'y/yes' to specify another filename containing the recorded waves
                              (must be binary NumPy '.npy' format). (Default)
                              Enter 'n/no' or 'q/quit to exit this program.
                              ''' %(dataFile)))
                    elif '.npy' not in dataFile:
                        print(textwrap.dedent(
                              '''
                              Error: File \'%s\' is not NumPy '.npy' format.
                              
                              Enter 'y/yes' to specify another filename containing the recorded waves
                              (must be binary NumPy '.npy' format). (Default)
                              Enter 'n/no' or 'q/quit to exit this program.
                              ''' %(dataFile)))
                elif answer == 'n' or answer == 'no' or answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.\n')                
                else:
                    print('Invalid response. Please enter \'y/yes\', \'n/no\', or \'q/quit\'.')
    
        #==============================================================================
        if Path(os.path.join(dataPath, 'impulseResponses.npy')).exists():
            impulseResponses = os.path.join(dataPath, 'impulseResponses.npy')
            noImpulseResponse = False
        else:
            userResponded = False
            print(textwrap.dedent(
                 '''
                 Warning: Expected file \'impulseResponses.npy\' not found. Does a file
                 exist containing the simulated impulse responses? (This is NOT a required file.)
                 
                 Enter 'y/yes' to specify the filename containing the simulated impulse responses
                 (must be binary NumPy '.npy' format).
                 Enter 'n/no' to proceed without specifying the impulse responses. (Default)
                 Enter 'q/quit' to exit this program.
                 '''))
            while userResponded == False:
                answer = input('Action: ')
                if answer == 'y' or answer == 'yes':
                    impulseResponsesFile = input('Please specify the filename containing the simulated impulse responses: ')
                    if '.npy' in impulseResponsesFile and Path(os.path.join(dataPath, impulseResponsesFile)).exists():
                        impulseResponses = os.path.join(dataPath, impulseResponsesFile)
                        noImpulseResponse = False
                        userResponded = True
                    elif '.npy' in impulseResponsesFile and not Path(os.path.join(dataPath, impulseResponsesFile)).exists():
                        print(textwrap.dedent(
                              '''
                              Error: File \'%s\' does not exist within the specified data directory.
                              
                              Enter 'y/yes' to specify another filename containing the simulated impulse responses
                              (must be binary NumPy '.npy' format).
                              Enter 'n/no' to proceed without specifying the impulse responses. (Default)
                              Enter 'q/quit' to exit this program.
                              ''' %(impulseResponsesFile)))
                    elif '.npy' not in impulseResponsesFile:
                        print(textwrap.dedent(
                              '''
                              Error: File \'%s\' is not NumPy '.npy' format.
                              
                              Enter 'y/yes' to specify another filename containing the simulated impulse responses
                              (must be binary NumPy '.npy' format).
                              Enter 'n/no' to proceed without specifying the impulse responses. (Default)
                              Enter 'q/quit' to exit this program.
                              ''' %(impulseResponsesFile)))
                elif answer == '' or answer == 'n' or answer == 'no':
                    print('Proceeding without specifying the simulated impulse responses.')
                    noImpulseResponse = True
                    userResponded = True
                elif answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.\n')
                else:
                    print('Invalid response. Please enter \'y/yes\', \'n/no\', or \'q/quit\'.')
    
        #==============================================================================
        # If a user is supplying impulse responses, the search points from which those
        # impulse responses were generated must also be supplied.
        if noImpulseResponse == False:
            if Path(os.path.join(dataPath, 'searchGrid.npz')).exists():
                searchGrid = os.path.join(dataPath, 'searchGrid.npz')
            else:
                userResponded = False
                print(textwrap.dedent(
                     '''
                     Error: Expected file \'searchGrid.npz\' not found. Does a file
                     exist containing the space-time search grid? (This file is
                     required only if impulse responses are provided.)
                         
                     Enter 'y/yes' to specify the filename containing the search grid
                     (must be binary NumPy '.npz' format). (Default)
                     Enter 'n/no' or 'q/quit' to exit this program.
                     '''))
                while userResponded == False:
                    answer = input('Action: ')
                    if answer == '' or answer == 'y' or answer == 'yes':
                        searchGridFile = input('Please specify the filename containing the search grid: ')
                        if '.npz' in searchGridFile and Path(os.path.join(dataPath, searchGridFile)).exists():
                            searchGrid = os.path.join(dataPath, searchGridFile)
                            userResponded = True
                        elif '.npz' in searchGridFile and not Path(os.path.join(dataPath, searchGridFile)).exists():
                            print(textwrap.dedent(
                                  '''
                                  Error: File \'%s\' does not exist within the specified data directory.
                                  
                                  Enter 'y/yes' to specify another filename containing the search grid
                                  (must be binary NumPy '.npz' format). (Default)
                                  Enter 'n/no' or 'q/quit' to exit this program.
                                  ''' %(searchGridFile)))
                        elif '.npz' not in searchGridFile:
                            print(textwrap.dedent(
                                  '''
                                  Error: File \'%s\' is not NumPy '.npz' format.
                                  
                                  Enter 'y/yes' to specify another filename containing the search grid
                                  (must be binary NumPy '.npz' format). (Default)
                                  Enter 'n/no' or 'q/quit' to exit this program.
                                  ''' %(searchGridFile)))
                    elif answer == 'n' or answer == 'no' or answer == 'q' or answer == 'quit':
                        sys.exit('Exiting program.\n')
                    else:
                        print('Invalid response. Please enter \'y/yes\', \'n/no\', or \'q/quit\'.')
        
        #==============================================================================  
        if noSources and noScatterer and noImpulseResponse:
            files = [receivers, recordingTimes, recordedData]
            np.savez('datadir.npz',
                     path = dataPath,
                     files = files,
                     receivers = receivers,
                     recordingTimes = recordingTimes,
                     recordedData = recordedData)
        elif noSources and noScatterer and not noImpulseResponse:
            files = [receivers, recordingTimes, recordedData, impulseResponses, searchGrid]
            np.savez('datadir.npz',
                     path = dataPath,
                     files = files,
                     receivers = receivers,
                     recordingTimes = recordingTimes,
                     recordedData = recordedData,
                     impulseResponses = impulseResponses,
                     searchGrid = searchGrid)
        elif noSources and not noScatterer and noImpulseResponse:
            files = [receivers, scatterer, recordingTimes, recordedData]
            np.savez('datadir.npz',
                     path = dataPath,
                     files = files,
                     receivers = receivers,
                     scatterer = scatterer,
                     recordingTimes = recordingTimes,
                     recordedData = recordedData)
        elif not noSources and noScatterer and noImpulseResponse:
            files = [receivers, sources, recordingTimes, recordedData]
            np.savez('datadir.npz',
                     path = dataPath,
                     files = files,
                     receivers = receivers,
                     sources = sources,
                     recordingTimes = recordingTimes,
                     recordedData = recordedData)
        elif noSources and not noScatterer and not noImpulseResponse:
            files = [receivers, scatterer, recordingTimes, recordedData, impulseResponses, searchGrid]
            np.savez('datadir.npz',
                     path = dataPath,
                     files = files,
                     receivers = receivers,
                     scatterer = scatterer,
                     recordingTimes = recordingTimes,
                     recordedData = recordedData,
                     impulseResponses = impulseResponses,
                     searchGrid = searchGrid)
        elif not noSources and noScatterer and not noImpulseResponse:
            files = [receivers, sources, recordingTimes, recordedData, impulseResponses, searchGrid]
            np.savez('datadir.npz',
                     path = dataPath,
                     files = files,
                     receivers = receivers,
                     sources = sources,
                     recordingTimes = recordingTimes,
                     recordedData = recordedData,
                     impulseResponses = impulseResponses,
                     searchGrid = searchGrid)
        elif not noSources and not noScatterer and noImpulseResponse:
            files = [receivers, sources, scatterer, recordingTimes, recordedData]
            np.savez('datadir.npz',
                     path = dataPath,
                     files = files,
                     receivers = receivers,
                     sources = sources,
                     scatterer = scatterer,
                     recordingTimes = recordingTimes,
                     recordedData = recordedData)
        else:
            files = [receivers, sources, scatterer, recordingTimes, recordedData, impulseResponses, searchGrid]
            np.savez('datadir.npz',
                     path = dataPath,
                     files = files,
                     receivers = receivers,
                     sources = sources,
                     scatterer = scatterer,
                     recordingTimes = recordingTimes,
                     recordedData = recordedData,
                     impulseResponses = impulseResponses,
                     searchGrid = searchGrid)