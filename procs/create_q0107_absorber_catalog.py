from barak.pyvpfit import readf26
from astro.io import writetxt

q0107a = readf26('/home/ntejos/catalogs/Q0107/A.f26')
q0107b = readf26('/home/ntejos/catalogs/Q0107/B.f26')
q0107c = readf26('/home/ntejos/catalogs/Q0107/C.f26')

cond = q0107a.lines.name == 'H I   '
a = q0107a.lines[cond]
cond = q0107b.lines.name == 'H I   '
b = q0107b.lines[cond]
cond = q0107c.lines.name == 'H I   '
c = q0107c.lines[cond]

a.name[:] = 'HI'
b.name[:] = 'HI'
c.name[:] = 'HI'

writetxt('/home/ntejos/catalogs/Q0107/A_HI.txt',[a.name,a.z,a.zsig,a.b,a.bsig,a.logN,a.logNsig])
writetxt('/home/ntejos/catalogs/Q0107/B_HI.txt',[b.name,b.z,b.zsig,b.b,b.bsig,b.logN,b.logNsig])
writetxt('/home/ntejos/catalogs/Q0107/C_HI.txt',[c.name,c.z,c.zsig,c.b,c.bsig,c.logN,c.logNsig])
