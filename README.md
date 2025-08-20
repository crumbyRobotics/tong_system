# tong_system
[Intelligent Systems and Informatics Laboratory](https://www.isi.imi.i.u-tokyo.ac.jp/?lang=ja), The University of Tokyo

<img src="image.jpg" alt="image" height="300"/>
<img src="image2.png" alt="image" height="300"/>




<br>


API of ISI's robot manipulator system called "Tong system".

- This package can be used as an interface to both the real and simulated robot.

The simulator is available at **https://github.com/crumbyRobotics/tong_simulator**.

## Install
* python >= 3.9

### PIP
Install packages
```
pip install -r requirements.txt 
```

Install tongsystem
```
pip install -e .
```

### Build Ikfastpy
Install libraries needed to build ikfastpy
```
sudo apt-get install liblapack-dev
sudo apt-get install liblapack3
sudo apt-get install libopenblas-base
sudo apt-get install libopenblas-dev
```

Build ikfastpy
```
git submodule update --init --recursive
cd tongsystem/ikfastpy
python setup.py build_ext --inplace
```

## Run with tongsim
1. git clone [tong_simulator repository](https://github.com/crumbyRobotics/tong_simulator) & setup following its README.
2. Launch tongsim in terminal 1: 
    ```
    cd tong_simulator
    python main.py
    ```
3. Run the example program in terminal 2:
    ```
    cd tong_system/tests
    python test_sim.py
    ```