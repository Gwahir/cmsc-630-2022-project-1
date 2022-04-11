import numpy as np
import numpy.lib.stride_tricks as slide

ERROR = 0.05
OPTIMIZE_INT = True
ALLOW_THREADS = 1

CLASSES = [
    'cyl',
    'inter',
    'let',
    'mod',
    'para',
    'super',
    'svar'
]

def rgbToMean(input, output):
    np.copyto(output, np.sum(input, axis=2) // 3, casting='unsafe')

def rgbToLightness(input, output):
    a = np.max(input, axis=2)
    b = np.min(input, axis=2)
    return (a + b) // 2
    
lumWeights = np.array([21, 72, 7])
def rgbToLuminosity(input, output):
    np.copyto(output, np.sum((lumWeights * input) // 100, axis=2), casting='unsafe')
    
def rgbToIntensity(input, output, weights, divisor = 1):
    weights = np.array(weights)
    if weights.shape != (0,) and len(weights.shape) == 1 and weights.shape[0] == 3:
        raise ValueError('Weights must be of shape [3]')
        
    np.copyto(output, np.sum((weights * input), axis=2) // 1, casting='unsafe')
    
def intensityToRgb(input, output):
    np.copyto(output, np.repeat(input.reshape(input.shape + (1,)), 3, axis=2), casting='unsafe')

def saltAndPepper(input, output, threshold):
    noise = np.random.rand(*input.shape)
    white = np.ones(input.shape) * 255
    black = np.zeros(input.shape)
    halfThreshold = threshold/2.0
    np.copyto(output, np.choose((noise < threshold) * 1 + (noise < halfThreshold) * 1, [input, white, black]), casting='unsafe')
    
def gaussianNoise(input, output, stdDeviation = 50):
    noise = np.random.normal(0, stdDeviation, input.shape)
    mixed = np.clip(np.random.normal(0,stdDeviation,input.shape) + input, 0, 255, dtype="uint8", casting="unsafe")
    np.copyto(output, mixed, casting='unsafe')
    
def noise(output):
    noise = np.random.rand(*output.shape)
    np.copyto(output, noise*255, casting='unsafe')
    
def histogram(input, output):
    step = 255 / len(output)
    output.fill(0)
    # Implementation that doesn't use specific method from library as demonstration of knowledge:
    # indices = np.array([intensity // step for y in input for intensity in y], dtype='uint')
    # for i in indices:
    #    histo[i] = histo[i] + 1
    gram, _ = np.histogram(input, bins=output.shape[0])
    np.copyto(output, gram, casting='unsafe')
    
def quantize(input, output, ts):
    indices = np.digitize(input, ts, right=True) - 1
    levels = np.append(ts, 255)
    np.copyto(output, [[(levels[i + 1] - levels[i]) / 2 + levels[i]  for i in j] for j in indices], casting='unsafe')
        
def quantizeUniform(input, output, delta):
    quantize(input, output, np.arange(0,255,delta))
    #np.copyto(output, [[delta * math.floor(val / delta) + delta / 2 for val in y] for y in input], casting='unsafe')
    
    

def slideKernel(input, kernel, mode='edge'):
    if(kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0):
        raise ValueError('Only odd dimension kernels are supported')
    padded = np.pad(input, (kernel.shape[0] // 2, kernel.shape[1] // 2), mode=mode)
    return slide.sliding_window_view(padded, kernel.shape) * kernel
    
def smoothFilter(input, output, kernel, mode='edge'):
    if abs(np.sum(kernel) - 1) > ERROR:
       raise ValueError('Smooth kernel must sum to 1')
    slid = slideKernel(input, kernel, mode)
    result = np.sum(np.sum(slid, axis=3), axis=2)
    np.copyto(output, np.clip(result, 0, 255), casting='unsafe')
    
def diffFilter(input, output, kernel, mode='edge'):
    if abs(np.sum(kernel)) > ERROR:
       raise ValueError('Difference kernel must sum to 0')
    slid = slideKernel(input, kernel, mode)
    result = np.sum(np.sum(slid, axis=3), axis=2) + 128
    np.copyto(output, np.clip(result, 0, 255), casting='unsafe')
    
def linearFilter(input, output, kernel, mode='edge'):
    if np.any(kernel, where=kernel):
        diffFilter(input, output, kernel, mode)
    else:
        smoothFilter(input, output, kernel, mode)

def medianFilter(input, output, kernel, mode='edge'):
    slid = slideKernel(input, kernel, mode)
    np.copyto(output, [[np.median(x) for x in y] for y in slid], casting='unsafe')
    
    
def averageOfHistrograms(listOfHistograms):
    rotated = np.swapaxes(listOfHistograms, 0, 1)
    return np.average(rotated, axis=1)
    