import lasio
import numpy as np
from PyNPEFA import PyNPEFA

# Read data
y = lasio.read('D:\Erich\Random Project\Well_Data\SEMBAKUNG-5_ARCIND\Log Digital\WCL0001834.LAS').df().GR.dropna()
x = np.array(y.index.tolist())

# Apply PyNPEFA
inpefa_log = PyNPEFA(y,x)