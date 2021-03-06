# Code Overview
In this section the general structure of the code will be describe as well as the used frameworks. Additional information on the usage will be given as well. 

The code was written on Ubuntu 16.04 with Python Version 3.6.1. It uses *numpy*, *scipy* and *tqdm* to handle arrays, distances and progress information. Furthermore it uses *OpenCV* to read and write images as well as for resizing, Gaussian blur and image presentation. Everything except for *OpenCV* can be installed executing the command `pip3 install -r requirements.txt`.
This assumes that *pip* is installed and setup. Tutorials for that can be found online. 

To install *OpenCV* with Python 3 support it is currently required to build *OpenCV* from source. A detailed guide on this process can be found on the [pyimagesearch blog by Adrian Rosebrock](http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/) . 
On some systems there can be issues with the GCC version when building OpenCV, as version 5 is required and often version 6 is installed by default. The following command can be used to prefer GCC 5 over GCC 6. `sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 1`.

With all the requirements set up it is now possible to execute the code. All the methods are within a file called `mean_shift.py`. To execute it run `python3 mean_shift.py` This command will execute the mean shift algorithm on the passed image. To further control the algorithm additional parameters can be passed. Information about them can be shown by executing `python3 mean_shift.py --help`. 

The core of the algorithm are the methods *meanshift* and *find_peak*. They are available optimized and non-optimized, indicated by the suffix ‘_opt’. The *image_segment* methods resizes and blurs an input image and converts it colorspace for better results in mean-shift. It is advised to use this method. In addition to that the code contains three utility functions: *get_neighbours* *get_neigbours_cdist* and *generate_report*. The first two methods are used find points that are some distance away from another point. The ‘_cdist’ implementation calculates euclidean distances between all points, the other one uses binary trees in the form of kd-trees which proved faster for repetitive searches in the same data points. The generate_report method can be called passing --report to the command line interface and generates eighty images with different combinations of parameters.

## TL;DR

- Use current Python 3
- `pip install -r requirements.txt`
- [Install OpenCV for Python 3](http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/)
  - Issues with GCC 5 vs 6 ? Run : `sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 1`
- `python3 mean_shift.py [image]` to run and `python3 mean_shift.py --help` for additional information
