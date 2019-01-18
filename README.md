openGPC 
===

Implements the sparse stereo method *Global Patch Collider* by Shenlong Wang, Sean Ryan Fanello, 
Christoph Rhemann, Shahram Izadi, Pushmeet Kohli, CVPR 2016

# Requirements
- Eigen3
- libpng
- CMake >= 3.4
- Sintel Stereo or Optical Flow datasets (http://sintel.is.tue.mpg.de/stereo).
This is only required if you would like to retrain a forest with different parameters.
Example forests are provided in the repository in the `forests` directory.

# Build
This is a header-only library and does not require building. 
The following instructions show how to build the examples contained in the directory `samples`.
Assuming we are within the root directory of the cloned repository:

```
cd samples
mkdir build
cd build
cmake ..
make 
```
The source code heavily relies on SSE instructions, which are enabled by default.
To build on another target platform such as ARM, supply the `SSE=OFF` argument to cmake, i.e.
use `cmake -DSSE=OFF ..` instead of the above `cmake ..`.

# Downloading datasets
If you'd like to train with either of the Sintel datasets, please refer to
`downloadSintelOpticalFlow.sh` and `downloadSintelStereo.sh` in the `data` directory.
These scripts will download and unpack the respective datasets. Please note
that both of these downloads are large (2 and 5GB)

# Running the examples
All examples have default parameters and run out of the box. Upon calling
the executables without arguments, usage information is displayed.

- `extract`: Mines a dataset from the Sintel dataset and stores it in an intermediary 
binary format. **Requires** the OpticalFlow dataset to be present (see section above)
- `train`: Trains a forest based on the dataset mined with `extract`.
**Requires** a previously extracted dataset, produced by extract
- `sparsematch`: Sparse matching based on pretrained forest. Outputs disparity estimate
to the build directory.


## License 
This software is licensed under the BSD 3-Clause License 
(also see https://opensource.org/licenses/BSD-3-Clause) for **non-commercial use**:

    Copyright (c) 2018, ETH Zurich
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without modification, 
    are permitted provided that the following conditions are met:
    
    1. Redistributions of source code must retain the above copyright notice, this 
    list of conditions and the following disclaimer.
    
    2. Redistributions in binary form must reproduce the above copyright notice, 
    this list of conditions and the following disclaimer in the documentation and/or 
    other materials provided with the distribution.
    
    3. Neither the name of the copyright holder nor the names of its contributors 
    may be used to endorse or promote products derived from this software without 
    specific prior written permission.
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON 
    ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

