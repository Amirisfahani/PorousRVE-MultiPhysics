/dolfinx-env/bin/python /root/Model/rveElasticityDamage.py
/dolfinx-env/bin/python /root/Model/revElasticityDamage.py
head -50 /root/rve_output/rve_quad_mesh_Lx1.00_phi30_hydrogen_diffusion_damage_channels.xdmf
/dolfinx-env/bin/python /root/Model/coupling.py
source /dolfinx-env/bin/activate
/dolfinx-env/bin/python /root/Model/RVEModel.py
pip install meshio
/dolfinx-env/bin/python /root/Model/RVEModel.py
pip install h5py --no-binary=h5py
/dolfinx-env/bin/python /root/Model/RVEModel.py
/dolfinx-env/bin/python /root/Model/RVEm.py
/dolfinx-env/bin/python /root/Model/RVEModel.py
/dolfinx-env/bin/python /root/Model/RVEm.py
/dolfinx-env/bin/python /root/Model/rveElasticityDamage.py
/dolfinx-env/bin/python /root/Model/revElasticityDamage.py
/dolfinx-env/bin/python /root/Model/rveElastoplastic_J2.py
/dolfinx-env/bin/python /root/Model/hydrogenDiffusion.py
/dolfinx-env/bin/python /root/Model/diffusion.py
/dolfinx-env/bin/python /root/Model/coupling.py
/dolfinx-env/bin/python /root/Model/RVEm.py
/dolfinx-env/bin/python /root/Model/revElasticityDamage.py
