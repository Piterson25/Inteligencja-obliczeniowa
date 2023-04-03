import numpy as np


def fact(x):
    return 1 / (1 + np.exp(-x))


def forwardPass(wiek, waga, wzrost):
    hidden1 = (wiek * -0.46122 + waga * 0.97314 + wzrost * -0.39203)
    hidden1_po_aktywacji = fact(hidden1 + 0.80109)
    hidden2 = (wiek * 0.78548 + waga * 2.10584 + wzrost * -0.57847)
    hidden2_po_aktywacji = fact(hidden2 + 0.43529)

    output = (hidden1_po_aktywacji * -0.81546 + hidden2_po_aktywacji * 1.03775) + -0.2368
    return output


print(forwardPass(23, 75, 176))  # 0.7985341880063129
print(forwardPass(25, 67, 180))  # 0.8009499165011525
