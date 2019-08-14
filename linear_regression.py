from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
all_data = pd.read_csv('data/Admission_Predict.csv')
#all_data = pd.read_csv('data.csv')
def scale_up(dataarr, mindata, maxdata):
	new_data=[]
	for val in dataarr:
		new_data.append((val*(maxdata-mindata))+mindata)
	return new_data 

def scale_data(dataarr, mindata, maxdata):
	new_data=[]
	for val in dataarr:
		new_data.append((val-mindata)/(maxdata-mindata))
	return new_data 

x = scale_data(all_data['CGPA'], all_data['CGPA'].min(), all_data['CGPA'].max())
y = scale_data(all_data['admit_chance'], all_data['admit_chance'].min(), all_data['admit_chance'].max())

#Line of best fit
def return_error_of_line(m, c, xdata, ydata):
	error = 0
	for i in range(len(xdata)):
		x_coord=xdata[i]
		y_coord=ydata[i]
		error = error + (y - ((m*x)+c)) ** 2

	return error/len(xdata)

def gradient_line(start_m, start_c, num_iter, xdata, ydata):
	c = start_c
	m = start_m
	for i in range(num_iter):
		c, m = step_gradient(c, m, xdata, ydata)
		
	return [c, m]
def step_gradient(current_c, current_m, xdata, ydata):
	m_gradient = 0
	c_gradient = 0
	N = len(xdata)
	for i in range(len(xdata)):
		x = xdata[i]
		y = ydata[i]

		m_gradient = m_gradient - ((2/N)*(y-(current_m*x+current_c)))*x
		c_gradient = c_gradient - ((2/N)*(y-(current_m*x+current_c)))

	return [current_c - (0.001)*c_gradient, current_m - (0.001)*m_gradient]

new_c, new_m = gradient_line(0, 0, 100000, x, y)

line_best_fit_x=np.linspace(min(x), max(x), 100)
line_best_fit_y=new_m*line_best_fit_x+new_c
plt.scatter(x, y, color="green", label="cgpa_chance")
plt.plot(line_best_fit_x, line_best_fit_y)
plt.title("Chance of admission in a University")
plt.xlabel("CGPA")
plt.ylabel("Chance")
plt.show()




