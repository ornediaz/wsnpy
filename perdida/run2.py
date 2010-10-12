# Syntax:
# python run.py ProdGlb.graphRate1 15 1 1
import sys
import pdb
import subprocess
import platform

comando = sys.argv[1] + "(" + ', '.join(sys.argv[2:]) + ")"
print "comando = " + comando
filename = '_'.join(sys.argv[1:])
with open('net2.cs', 'r') as f1:
    l1 = f1.readlines()  
with open(filename + '.cs', 'w') as f2:
    for i, line in enumerate(l1):
        if 'TOREPLACE' in line:
            f2.write('comando = "{0}";\n'.format(comando))
            f2.write(comando + ";\n")
        else:
            f2.write(line)
if platform.system() == 'Windows':
    compilar = ["csc", "/debug+", "/warnaserror+", filename + ".cs"]
    print compilar
    subprocess.check_call(compilar)
    # subprocess.check_call(['{0}.exe'.format(filename),])
    subprocess.check_call(['mdbg', '{0}.exe'.format(filename)])
if platform.system() == "Linux":
    compilar = ["gmcs", filename + ".cs"]
    print compilar
    subprocess.check_call(compilar)
    subprocess.check_call(['mono', '{0}.exe'])
            
            
