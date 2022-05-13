import numpy as np
import numpy.lib.stride_tricks as slide
from scipy.ndimage.interpolation import shift
import math
import proj1 as filters

CLASSES = filters.CLASSES

def rgbToMean(input):
    return np.sum(input, axis=2) // 3
    

def rgbToLightness(input, output):
    a = np.max(input, axis=2)
    b = np.min(input, axis=2)
    return (a + b) // 2
    
lumWeights = np.array([21, 72, 7])
def rgbToLuminosity(input):
    return np.sum((lumWeights * input) // 100, axis=2)
    
def rgbToIntensity(input, weights, divisor = 1):
    weights = np.array(weights)
    if weights.shape != (0,) and len(weights.shape) == 1 and weights.shape[0] == 3:
        raise ValueError('Weights must be of shape [3]')
        
    return np.sum((weights * input), axis=2) // 1
    
def intensityToRgb(input):
    return np.repeat(input.reshape(input.shape + (1,)), 3, axis=2)

def saltAndPepper(input, threshold):
    noise = np.random.rand(*input.shape)
    white = np.ones(input.shape) * 255
    black = np.zeros(input.shape)
    halfThreshold = threshold/2.0
    return np.choose((noise < threshold) * 1 + (noise < halfThreshold) * 1, [input, white, black])
    
def gaussianNoise(input, stdDeviation = 50):
    noise = np.random.normal(0, stdDeviation, input.shape)
    mixed = np.clip(np.random.normal(0,stdDeviation,input.shape) + input, 0, 255, dtype="uint8", casting="unsafe")
    return mixed
    
def noise():
    noise = np.random.rand(*output.shape)
    return noise*255
    
def histogram(input):
    step = 255 / len(output)
    output.fill(0)
    gram, _ = np.histogram(input, bins=output.shape[0])
    return gram
    
def quantize(input, ts):
    indices = np.digitize(input, ts, right=True) - 1
    levels = np.append(ts, 255)
    return [[(levels[i + 1] - levels[i]) / 2 + levels[i]  for i in j] for j in indices]
        
def quantizeUniform(input, output, delta):
    return quantize(input, output, np.arange(0,255,delta))
    #np.copyto(output, [[delta * math.floor(val / delta) + delta / 2 for val in y] for y in input], casting='unsafe')
    
    

def slideKernel(input, kernel, mode='edge'):
    if(kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0):
        raise ValueError('Only odd dimension kernels are supported')
    padded = np.pad(input, (kernel.shape[0] // 2, kernel.shape[1] // 2), mode=mode)
    return slide.sliding_window_view(padded, kernel.shape) * kernel
    
def smoothFilter(input, kernel, mode='edge'):
    if abs(np.sum(kernel) - 1) > ERROR:
       raise ValueError('Smooth kernel must sum to 1')
    slid = slideKernel(input, kernel, mode)
    result = np.sum(np.sum(slid, axis=3), axis=2)
    return np.clip(result, 0, 255)
    
def diffFilter(input, kernel, mode='edge'):
    if abs(np.sum(kernel)) > ERROR:
       raise ValueError('Difference kernel must sum to 0')
    slid = slideKernel(input, kernel, mode)
    result = np.sum(np.sum(slid, axis=3), axis=2) + 128
    return np.clip(result, 0, 255)
    
def linearFilter(input, kernel, mode='edge'):
    if np.any(kernel, where=kernel):
        return diffFilter(input, kernel, mode)
    else:
        return smoothFilter(input, kernel, mode)

def medianFilter(input, kernel, mode='edge'):
    slid = slideKernel(input, kernel, mode)
    flat = input.reshape((input.shape[0], input.shape[1], input.shape[2] * input.shape[3]))
    return np.median(flat, axis=2)

# def oldMedianFilter(input, kernel, mode='edge'):
#     slid = slideKernel(input, kernel, mode)
#     return [[np.median(x) for x in y] for y in slid]
    
    
def averageOfHistrograms(listOfHistograms):
    rotated = np.swapaxes(listOfHistograms, 0, 1)
    return np.average(rotated, axis=1)


def toBoolean(input, threshold=10):
    return (1 - (input > threshold)).astype(np.byte)

def makeBooleanDisc(diameter):
    if diameter % 2 != 1:
        raise Error("diameter must be odd")
        
    radius = diameter // 2
    y,x = np.ogrid[-radius: radius+1, -radius: radius+1]
    mask = x**2+y**2 <= radius**2
    return mask.astype(np.byte)


impSobelX = np.asarray([
    [-3, 0, 3],
    [-10, 0, 10],
    [-3, 0, 3]
])
impSobelY = np.asarray([
    [-3, -10, -3],
    [0, 0, 0],
    [3, 10, 3]
])

def improvedSobel(input):
    slid = filters.slideKernel(input, impSobelX)*(1.0/32.0)
    edgeX = np.sum(np.sum(slid, axis=3), axis=2)
    slid = filters.slideKernel(input, impSobelY)*(1.0/32.0)
    edgeY = np.sum(np.sum(slid, axis=3), axis=2)

    return np.sqrt(edgeX ** 2 + edgeY **2)
    
    

def improvedSobelWithTheta(input):
    slid = filters.slideKernel(input, impSobelX)*(1.0/32.0)
    edgeX = np.sum(np.sum(slid, axis=3), axis=2)
    slid = filters.slideKernel(input, impSobelY)*(1.0/32.0)
    edgeY = np.sum(np.sum(slid, axis=3), axis=2)

    edge = np.sqrt(edgeX ** 2 + edgeY **2)
    theta = np.arctan(edgeY / (edgeX + 0.00001))
    return edge, theta



roberts1 = np.asarray([[1, 0, 0], [0, -1, 0], [0,0,0]])
roberts2 = np.asarray([[0, 1, 0], [-1, 0, 0], [0,0,0]])
def robertsEdge(input):
    slid = filters.slideKernel(input, roberts1)
    edgeX = np.sum(np.sum(slid, axis=3), axis=2)
    slid = filters.slideKernel(input, roberts2)
    edgeY = np.sum(np.sum(slid, axis=3), axis=2)
    edge = np.sqrt(edgeX ** 2 + edgeY **2)
    return edge
    
    
def robertsEdgeWithTheta(input):
    slid = filters.slideKernel(input, roberts1)
    edgeX = np.sum(np.sum(slid, axis=3), axis=2)
    slid = filters.slideKernel(input, roberts2)
    edgeY = np.sum(np.sum(slid, axis=3), axis=2)
    edge = np.sqrt(edgeX ** 2 + edgeY **2)
    theta = np.arctan(edgeY / (edgeX + 0.00001)) - 3*math.pi/4
    return edge, theta


compass0 = np.asarray([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]])
compass1 = np.asarray([
    [-2, -1, 0],
    [-1, 0, 1],
    [0, 1, 2]])
compass2 = np.asarray([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]])
compass3 = np.asarray([
    [0, -1,-2],
    [1, 0, -1],
    [2, 1, 0]])


compasses = [compass0, compass1, compass2, compass3, -compass0, -compass1, -compass2, -compass3]

def compassEdge(input):
    compassValues = [np.sum(np.sum(filters.slideKernel(input, compass), axis=3), axis=2) for compass in compasses]
    max = np.maximum(compassValues[0], compassValues[1])
    for i in range(2,8):
        max = np.maximum(max, compassValues[i])
    return max


def dilate(input, mask):
    offsets = np.asarray(mask.shape) // 2
    shifted = [mask[u,v] * shift(input, (u - offsets[0],v - offsets[1])) for v,y in enumerate(mask) for u,x in enumerate(y)]
    return np.logical_or.reduce(shifted).astype(np.byte)
    

def erode(input, mask, mode='edge'):
    slide = filters.slideKernel(input, mask, mode)
    return (np.sum(np.sum(slide, axis=3), axis=2) == np.sum(mask)).astype(np.byte)

def growAndShrink(input, mask, steps=1):
    morph = input
    for _ in range(steps):
        morph = dilate(morph, mask)
    for _ in range(steps):
        morph = erode(morph, mask)
    return morph


def shrinkAndGrow(input, mask, steps=1):
    morph = input
    for _ in range(steps):
        morph = erode(morph, mask)
    for _ in range(steps):
        morph = dilate(morph, mask)
    return morph

def thresholdSegmentation(input, threshold):
    return (input <= threshold).astype(np.byte)

def calcGroupVariance(input, threshold):
    segmented = (input <= threshold).astype(np.byte)
    
    objProbability = np.count_nonzero(segmented) / input.size
    
    if objProbability == 0:
        objVariance = 0
    else:
        objVariance = np.var(input, where = segmented == 1)
        
    if objProbability == 1:
        bakVariance = 0
    else:
        bakVariance = np.var(input, where = segmented == 0)
    
    return objProbability * objVariance + (1 - objProbability) * bakVariance
    
    

def findBestSegmentationThreshold(input, stepSize = 10):
    thresholds = range(stepSize, 255, stepSize)
    variances = [calcGroupVariance(input, threshold) for threshold in thresholds]
    return thresholds[np.argmin(variances)]
    
    
def autoThresholdSegmentation(input, searchStepSize = 10):
    return thresholdSegmentation(input, findBestSegmentationThreshold(input, searchStepSize))
    
    
def kMeansSegmentation(img, k):
    imgSet = np.reshape(img, (img.shape[0] * img.shape[1], 3))
    indices = [[x,y] for x in range(img.shape[0]) for y in range(img.shape[1])]
    #data = np.append(imgSet, indices, axis=1)
    data = imgSet
    vectorLength = data.shape[1]
    
    kPoints = np.asarray([
        [np.random.randint(0, 255) for _ in range(vectorLength)] for _ in range(k)])
    
    oldPoints = np.zeros(kPoints.shape)


    while not np.all(oldPoints == kPoints):
        assignments = np.argmin(np.linalg.norm(np.repeat(np.reshape(data, (data.shape[0], 1, vectorLength)), k, axis=1) - kPoints, axis=2), axis=1)
        ass = np.repeat(np.reshape(assignments, (data.shape[0], 1)), vectorLength, axis=1)
        oldPoints = kPoints
        kPoints = np.round([np.mean(data, where = ass == kIndex * np.ones(vectorLength), axis=0) for kIndex in range(k)]).astype(int)

    
    return np.asarray(np.split(assignments, img.shape[0]))