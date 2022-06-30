import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pandas as pd

NUM_SAMPLES = 100
DOMAIN = 2 * np.pi

def squareWave(x, amplitude, frequency, phase):
    x = x + phase
    return amplitude * np.sign( np.sin(2 * np.pi * frequency * x) )

def triangleWave(x, amplitude, period, phase):
    x= x + phase
    a = 4 * amplitude / period
    return a * np.abs( ((x - period/4) % period) - period/2 ) - amplitude

def sineWave(x, amplitude, frequency, phase):
    return amplitude * np.sin(2 * np.pi * x * frequency + phase)

def addNoise(wave, d = 0.1):
    wave += np.random.uniform(-d, d, size=NUM_SAMPLES)

def generateWaves(num, generator, foldername, filenameFormat, noise = 0.1, ampRange = (-3.0, 3.0), freqRange = (0.5, 2.5), phaseRange = (-1.0, 1.0)):
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    for i in range(num):
        xValues = np.random.uniform(0, 1, size=NUM_SAMPLES)
        xValues.sort()
        amp, freq, phase = random.uniform(*ampRange), random.uniform(*freqRange), random.uniform(*phaseRange)
        yValues = generator(xValues, amp, freq, phase)
        yValues += np.random.uniform(-noise, noise, size=NUM_SAMPLES)

        filename = filenameFormat.format(i)
        filepath = os.path.join(foldername, filename)
        df = pd.DataFrame({ "x": xValues, "y": yValues })
        df.to_csv(filepath, index=False)

def viewFile(filepath):
    df = pd.read_csv(filepath)
    xValues = df["x"]
    yValues = df["y"]

    plt.plot(xValues, yValues, "r.")
    plt.show()

if __name__ == "__main__":
    generateWaves(200, sineWave, "data/sine", "sine_{}.csv")
    generateWaves(200, squareWave, "data/square", "square_{}.csv")
    generateWaves(200, triangleWave, "data/triangle", "tri_{}.csv")
    viewFile("data/triangle/tri_0.csv")

