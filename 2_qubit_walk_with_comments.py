import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy as sp
import time
mpl.use('TkAgg')

H = np.loadtxt('adjmat.txt') # import adjacency matrix from text file

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import BasicAer, execute
from qiskit.quantum_info.synthesis import two_qubit_cnot_decompose
backend = BasicAer.get_backend('qasm_simulator')

steps = 200 # number of time steps
ts = np.linspace(0,10,steps) # create list of discrete time values to use in loop
shots = 1000 # number of simulations for each time step

ys = {'00':[],'01':[],'10':[],'11':[]} # dictionary to hold counts for quantum simulation
yse = {'00':[],'01':[],'10':[],'11':[]} # dictionary to hold classically calculated values for comparison

start = time.time()
for i,t in enumerate(ts):

	M = sp.linalg.expm(-1j*H*t) # create time evolution operator
	st = M.dot(np.array([1,0,0,0])) # apply operator to vacuum state
	cr = ClassicalRegister(2,'c'+str(i)) # create and name classical register
	qc = two_qubit_cnot_decompose(M) # decompose time evolution operator into quantum gates
	qc.add_register(cr) # add classical register to circuit
	qc.measure([0,1],[0,1]) # add measurement gate
	result = execute(qc, backend,shots=shots).result() # simulate
	counts = result.get_counts(qc) # get results
	for i,state in enumerate(ys.keys()): # populate dictionaries
		yse[state].append(np.linalg.norm(st[i])**2)
		try:
			ys[state].append(counts[state]/shots)
		except KeyError:
			ys[state].append(0)

print("Quantum walk finished in {0} seconds.".format(time.time()-start))
	
qc.draw(output='latex',plot_barriers=False,filename='2q_walk_circuit.png') # draw circuit

plt.figure(1) # plot figure

for state in ys.keys():
	plt.plot(ts,ys[state],'o',label=state)
	plt.plot(ts,yse[state],'-k')

plt.xlabel('Time (seconds)')
plt.ylabel('Probability')
plt.legend()
plt.title('4-State Quantum Walk, {0:d} Steps, {1:d} Shots'.format(steps,shots))
plt.savefig('2qwalk_py.png')
plt.show()