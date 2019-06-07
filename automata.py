import numpy as np
import sys
from numpy.random import randint
from time import sleep, time
from scipy.ndimage.filters import convolve

import pyqtgraph as pg
import pyqtgraph.opengl as gl

automaton_dir = 'automaton_examples/'

load_automaton = False
if len(sys.argv) > 1:
    load_automaton = True

    # prepend directory to filename
    if automaton_dir not in sys.argv[1]:
        sys.argv[1] = automaton_dir + sys.argv[1]

    life = np.load(sys.argv[1])
    print(f'loaded {life.sum()} active cells')

n = 50
assert n % 2 == 0, 'must be even'
delay = .4

if not load_automaton:
    life = randint(0,2,size=(n,n,n,))
    life = life & randint(0,2,size=(n,n,n,))
    life = np.zeros((n,n,n))
    life[:,0,0] = 1
    life[0,:,0] = 1
    life[0,0,:] = 1

## build a QApplication before building other widgets
pg.mkQApp()

## make a widget for displaying 3D objects
view = gl.GLViewWidget()
#view.showFullScreen()

## create three grids, add each to the view
xgrid = gl.GLGridItem()
ygrid = gl.GLGridItem()
zgrid = gl.GLGridItem()
view.addItem(xgrid)
view.addItem(ygrid)
view.addItem(zgrid)

## rotate x and y grids to face the correct direction
xgrid.rotate(90, 0, 1, 0)
ygrid.rotate(90, 1, 0, 0)

## scale each grid differently
xgrid.scale(0.2, 0.2, 0.2)
ygrid.scale(0.2, 0.2, 0.2)
zgrid.scale(0.2, 0.2, 0.2)

## place points
grid = gl.GLScatterPlotItem()
view.addItem(grid)

pos = np.empty((n,n,n,3))
for i in range(-n//2,n//2):
    pos [i,:,:,0] = i/(n/4)
    pos [:,i,:,1] = i/(n/4)
    pos [:,:,i,2] = i/(n/4)

filter_ = np.ones((3,3,3))
_life_flag = -100000 # a little hacky, but marks cell as alive or dead
filter_[1,1,1] = _life_flag # mark cell as alive or dead
#filter_[0,:,:]

def evaluate(
        cell,size,
        life_size = 16,
        dead_size = 0,
        starvation_lim = 2,
        overcrowded_lim = 3,
        reproduction_val = 3,
        ):
    if cell < 0: # l was alive
        cell -= _life_flag
        if cell < starvation_lim or cell > overcrowded_lim: # under/over-crowed
            cell = 0
            size = dead_size
        else:
            cell = 1
            size = life_size
    else: # l was dead
        if cell == reproduction_val: # reproduction
            cell = 1
            size = life_size
        else:      # else still dead
            cell = 0
            size = dead_size
    return cell,size
evaluate = np.vectorize(evaluate)

def next_generation(life, size):
    life = convolve(life, filter_, mode='wrap')
    #life = convolve(life, filter_, mode='constant', cval=1.0)
    #life = convolve(life, filter_, mode='constant', cval=0.0)
    #life = convolve(life, filter_)
    return evaluate(life,size)

def rand_color_tuple():
    color = np.random.rand(4)
    color[3] = 1 # last value is opacity (turned on)
    color[randint(0,3)] = 1 # at least one color on
    return tuple(color)

def run_simulation(
        life,
        delay = .4,
        breed = False,
        k = 3,
        min_life = 3,
        max_life = 32,
        gen_allowance = 4,
        open_window = "max",
        orbit_speed = .1,
        ):
    color = (1,1,1,1) # all on / white
    size = np.zeros(life.shape)
    time_last = time()
    gen_counter = 0

    if not open_window: delay = 0

    if open_window == "full": view.showFullScreen()
    elif open_window == "max": view.showMaximized()
    elif open_window:         view.show()

    while not view.isHidden():
        s = life.sum()
        gen_counter += 1

        if breed and \
                (s == 0 or gen_counter > gen_allowance) and \
                (s < min_life or s > max_life):
            gen_counter = 0

            color = rand_color_tuple()

            life[...] = 0
            life[0:k,0:k,0:k] = randint(0,2,size=(k,k,k)) \
                                * randint(0,2,size=(k,k,k)) \
                                * randint(0,2,size=(k,k,k)) \
                                * randint(0,2,size=(k,k,k)) \

        if time_last < time() - delay:
            time_last = time()
            life,size = next_generation(life,size)
            grid.setData(pos=pos.reshape(-1,3), color=color, size=size.flatten())
            #color = rand(n**3,4)
            #grid.setData(pos=pos.reshape(-1,3), color=color, size=16)

        view.orbit(orbit_speed,0)
        pg.QtGui.QApplication.processEvents()

    return life

if __name__ == "__main__":
    print('number of active cells:',life.sum())
    if load_automaton:
        life = run_simulation(life)
    else:
        life = run_simulation(
                    life,
                    k=3,
                    breed=True,
                    delay=.0,
                    gen_allowance=45,
                    min_life=12,
                    max_life=512,
                    )

    print('number of active cells:',life.sum())
    print('save automaton? [y/n]',end=' ')
    save = input().strip().lower()
    if save == 'y':
        filename = automaton_dir + f'automaton{round(time())}'
        np.save(filename, life)
        print(f'saved {filename}.npy!')
