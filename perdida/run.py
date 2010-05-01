import sys
import pdb
import subprocess
import platform

if len(sys.argv) < 4:
    raise Exception("Parameters: class.fun_name, tst_nr, n_averages")

command = sys.argv[1] + "(" + ', '.join(sys.argv[2:]) + ")"
print "command = " + command
filename = "{0}_{1:02d}_{2:06d}_".format(sys.argv[1], int(sys.argv[2]),
        int(sys.argv[3]))
if (len(sys.argv) > 4):
    filename += '_'.join(sys.argv[4:])
with open('net.cs', 'r') as f1:
    l1 = f1.readlines()  
with open(filename + '.cs', 'w') as f2:
    for i, line in enumerate(l1):
        if 'TOREPLACE' in line:
            f2.write(command + ";\n")
            f2.write('comando = "{0}";\n'.format(command))
        else:
            f2.write(line)
if platform.system() == 'Windows':
    compilar = ["csc", "/debug+", filename + ".cs"]
    print compilar
    subprocess.check_call(compilar)
    subprocess.check_call(['{0}.exe'.format(filename),])
if platform.system() == "Linux":
    compilar = ["gmcs", filename + ".cs"])
    print compilar
    subprocess.check_call(compilar)
    subprocess.check_call(['mono', '{0}.exe'])
            
            
