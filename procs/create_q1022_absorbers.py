from barak.pyvpfit import readf26
from astro.io import writetxt

q1022 = readf26('/home/ntejos/COS/q1022/FUV/q1022_fuv_all.f26')

cond = q1022.lines.name == 'H I   '
a = q1022.lines[cond]

a.name[:] = 'HI'

writetxt('/home/ntejos/catalogs/Q1022/HI.txt',[a.name,a.z,a.zsig,a.b,a.bsig,a.logN,a.logNsig])
