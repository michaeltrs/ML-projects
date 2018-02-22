from scipy.io import loadmat
import pickle
from helper_functions import *

# --------------------------------------------------------------- #
# USER INPUT

# Please provide full path to test data - .mat format assumed
data_file = 'Data/cleandata_students.mat'

# Option 1 - Test Data include both x and y
data = loadmat(data_file)
xtest = data['x']

# Option 2 - Test Data include only x
# xtest = loadmat(data_file)

# USER INPUT - END
# --------------------------------------------------------------- #

# Load trees
trees = pickle.load(open("trained_trees_clean_data.p", "rb" ))

# Make prediction based on trees
y_predicted = testTrees2(trees, xtest, 'performance')

# Save predictions to disk
np.savetxt('test_results.txt', y_predicted)
