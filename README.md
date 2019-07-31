## Installation and usage

- Python 3.7

- [VirtualEnv](https://virtualenv.pypa.io/en/stable/installation/) or Anaconda unless you want to install the required packages within your system's Python. `sudo apt-get install virtualenv` on Ubuntu.

- a `C++`-Compiler for PyStan. On Ubuntu, run `sudo apt-get install build-essential`. On Windows, follow [these instructions](https://pystan.readthedocs.io/en/latest/windows.html).



#### Installation

Run the following commands or perform the equivalent operations on Windows

```bash
# download and unpack code
git clone https://github.com/LyteFM/spatial-tobit.git 
# change working directory into unpacked folder
cd spatial-tobit
# create and activate your virtual/ conda environment 
virtualenv --python=python3.7 venv
source venv/bin/activate
# install the required python dependencies
pip install -r requirements.txt
```

#### Running

If you want to run less than the 30000 iterations that we ran to obtain our results, adjust the numbers for `iter` and `warmup` in the last line of `import_data.py` accordingly. Then, run the file e.g. within Spyder or from the command line with:

```bash
python3 import_data.py
```

If you wish to compare the runtimes instead of running the two most efficient implementations, comment and uncomment the respective lines at the and of the file and save the output. 

The models are cached into the `cache` folder and will be loaded again on subsequent runs with the same `iter` and `warmup`. The parameter results are saved into the `logs` folder.

With an Arggis Pro installation and the `arcpy` package, you can also do the geospatial mapping with `geoprocessing_NewYork.py`. Some more example models on a test dataset can be explored in `pystan_example.py`. 

# 


