import numpy as np
import tensornetwork
import opt_einsum
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams["font.size"] = 26


D_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
tn_times = []
oe_times = []

for D in D_list:
  x = np.ones(3 * (D,))
  y = np.ones(3 * (D,))
  z = np.ones(3 * (D,))
  w = np.ones(3 * (D,))

  net = tensornetwork.TensorNetwork()
  xn = net.add_node(x, axis_names=["b", "c", "a"])
  yn = net.add_node(y, axis_names=["c", "d", "b"])
  zn = net.add_node(z, axis_names=["d", "b", "f"])
  wn = net.add_node(w, axis_names=["a", "f", "c"])

  xn["a"] ^ wn["a"]
  yn["d"] ^ zn["d"]
  zn["f"] ^ wn["f"]


  # Copy node 1
  a = net.add_copy_node(rank=3, dimension=D)
  xn["b"] ^ a[0]
  yn["b"] ^ a[1]
  zn["b"] ^ a[2]

  # Copy node 2
  b = net.add_copy_node(rank=3, dimension=D)
  xn["c"] ^ b[0]
  yn["c"] ^ b[1]
  wn["c"] ^ b[2]

  start_time = time.time()
  net = tensornetwork.contractors.optimal(net)
  print(net.get_final_node().tensor)
  tn_times.append(time.time() - start_time)


  start_time = time.time()
  print(opt_einsum.contract("bca,cdb,dbf,afc", x, y, z, w))
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
plt.savefig("times_with_copies.pdf", bbox_inches="tight")

#alg = opt_einsum.paths.optimal
#res, nodes = tensornetwork.contractors.opt_einsum_paths.utils.get_path(net, alg)

#input_sets = [set(x for x in letters) for letters in "bca,cdb,dbf,afc".split(",")]
#size_dict = {x: 4 for inp in input_sets for x in inp}
#res2 = alg(input_sets, set(), size_dict)
