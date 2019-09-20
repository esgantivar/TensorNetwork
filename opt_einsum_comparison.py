import numpy as np
import tensornetwork
import opt_einsum
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams["font.size"] = 26


D_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
tn_times = []
oe_times = []

for D in D_list:
  x = np.ones(3 * (D,))
  y = np.ones(3 * (D,))
  z = np.ones(3 * (D,))
  w = np.ones(3 * (D,))

  net = tensornetwork.TensorNetwork()
  xn = net.add_node(x, axis_names=["b", "c", "a"])
  yn = net.add_node(y, axis_names=["c", "d", "g"])
  zn = net.add_node(z, axis_names=["d", "b", "f"])
  wn = net.add_node(w, axis_names=["a", "f", "g"])

  xn["a"] ^ wn["a"]
  xn["b"] ^ zn["b"]
  xn["c"] ^ yn["c"]
  yn["d"] ^ zn["d"]
  zn["f"] ^ wn["f"]
  yn["g"] ^ wn["g"]


  start_time = time.time()
  net = tensornetwork.contractors.optimal(net)
  print(net.get_final_node().tensor)
  tn_times.append(time.time() - start_time)


  start_time = time.time()
  print(opt_einsum.contract("bca,cdg,dbf,afg", x, y, z, w, optimize="optimal"))
  oe_times.append(time.time() - start_time)


plt.figure(figsize=(7, 4))
plt.plot(D_list, tn_times, color="red", marker="o", linewidth=2.5,
         markersize=8, label="TensorNetwork")
plt.plot(D_list, oe_times, color="blue", marker="d", linewidth=2.5,
         markersize=8, label="opt_einsum")
plt.legend()
plt.xlabel("$\chi$")
plt.ylabel("Time (sec)")
#plt.show()
plt.savefig("times_without_copies.pdf", bbox_inches="tight")
