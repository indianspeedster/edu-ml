::: {.cell .markdown}
# Run a single user notebook server on Chameleon

This notebook describes how to run a single user Jupyter notebook server on Chameleon. This allows you to run experiments requiring bare metal access, storage, memory, GPU and compute resources on Chameleon using a Jupyter notebook interface.

> This notebook assumes that you already have completed [Hello, Chameleon](https://github.com/teaching-on-testbeds/hello-chameleon/blob/main/hello_chameleon.ipynb)

:::

::: {.cell .markdown}
## Provision the resource
:::

::: {.cell .markdown}
### Check resource availability
:::

::: {.cell .markdown}
This notebook will try to reserve a bare metal Ubuntu on <CHI@UC> - pending availability. Before you begin, you should check the host calendar at <https://chi.uc.chameleoncloud.org/project/leases/calendar/host/> to see what node types are available.
:::

::: {.cell .markdown}
### Chameleon configuration

You can change your Chameleon project name (if not using the one that is automatically configured in the JupyterHub environment) and the site on which to reserve resources (depending on availability) in the following cell.
:::

::: {.cell .code}
``` python
import chi, os

PROJECT_NAME = os.getenv('OS_PROJECT_NAME')
chi.use_site("CHI@UC")
chi.set("project_name", PROJECT_NAME)
```
:::

::: {.cell .markdown}
If you need to change the details of the Chameleon server, e.g. use a different OS image, or a different node type depending on availability, you can do that in the following cell.

:::

::: {.cell .code}
``` python
chi.set("image", "CC-Ubuntu20.04")
NODE_TYPE = "gpu_rtx_6000"
```
:::

::: {.cell .markdown}
### Reservation

The following cell will create a reservation that begins now, and ends in 8 hours. You can modify the start and end date as needed.
:::

::: {.cell .code}
``` python
from chi import lease


res = []
lease.add_node_reservation(res, node_type=NODE_TYPE, count=1)
lease.add_fip_reservation(res, count=1)
start_date, end_date = lease.lease_duration(days=0, hours=8)

l = lease.create_lease(f"{os.getenv('USER')}-edu-ml", res, start_date=start_date, end_date=end_date)
l = lease.wait_for_active(l["id"])
```
:::

::: {.cell .markdown}
### Provisioning resources

This cell provisions resources. It will take approximately 10 minutes. You can check on its status in the Chameleon web-based UI: <https://chi.uc.chameleoncloud.org/project/instances/>, then come back here when it is in the READY state.
:::

::: {.cell .code}
``` python
from chi import server

reservation_id = lease.get_node_reservation(l["id"])
server.create_server(
    f"{os.getenv('USER')}-edu-ml", 
    reservation_id=reservation_id,
    image_name=chi.get("image")
)
server_id = server.get_server_id(f"{os.getenv('USER')}-edu-ml")
server.wait_for_active(server_id)
```
:::

::: {.cell .markdown}
Associate an IP address with this server:
:::

::: {.cell .code}
``` python
reserved_fip = lease.get_reserved_floating_ips(l["id"])[0]
server.associate_floating_ip(server_id,reserved_fip)
```
:::

::: {.cell .markdown}
and wait for it to come up:
:::

::: {.cell .code}
``` python
server.wait_for_tcp(reserved_fip, port=22)
```
:::

::: {.cell .markdown}
## Install stuff
:::

::: {.cell .markdown}
The following cells will install some basic packages on your Chameleon server.
:::

::: {.cell .code}
``` python
from chi import ssh

node = ssh.Remote(reserved_fip)
```
:::

::: {.cell .code}
``` python
node.run('sudo apt update')
node.run('sudo apt -y install python3-pip python3-dev')
node.run('sudo pip3 install --upgrade pip')
```
:::

::: {.cell .markdown}
### Install CUDA and Nvidia Drivers
:::

::: {.cell .code}
``` python
node.run('wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb')
node.run('sudo dpkg -i cuda-keyring_1.0-1_all.deb')
node.run('sudo apt update')
node.run('sudo apt -y install linux-headers-$(uname -r)')
node.run('sudo apt-mark hold cuda-toolkit-12-config-common nvidia-driver-535') 
node.run('sudo apt -y install nvidia-driver-520') 
```
:::

::: {.cell .code}
```python
try:
    node.run('sudo reboot') # reboot and wait for it to come up
except:
    pass
server.wait_for_tcp(reserved_fip, port=22)
node = ssh.Remote(reserved_fip) 

```

:::

::: {.cell .code}
```python
node.run('sudo apt -y install cuda-11-8 cuda-runtime-11-8 cuda-drivers=520.61.05-1')
node.run('sudo apt -y install nvidia-gds-11-8') 
node.run('sudo apt -y install libcudnn8=8.9.3.28-1+cuda11.8 nvidia-cuda-toolkit') 

```

:::

::: {.cell .code}
```python
node.run("echo 'PATH=\"/usr/local/cuda-11.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin\"' | sudo tee /etc/environment")

```

:::

:::{.cell .markdown}
Now we have to reboot, and make sure we have the specified CUDA:

:::

::: {.cell .code}
```python
try:
    node.run('sudo reboot')
except:
    pass
server.wait_for_tcp(reserved_fip, port=22)
node = ssh.Remote(reserved_fip) # note: need a new SSH session to get new PATH
node.run('nvidia-smi')
node.run('nvcc --version')

```

:::

::: {.cell .markdown}
For this sequence of notebooks, we will require `pytorch`:
:::

::: {.cell .code}
``` python
node.run('python3 -m pip install --user torch==2.0.0')
```
:::

::: {.cell .code}
``` python
node.run('python3 -m pip install --user torchvision==0.15.1')
```
:::

::: {.cell .code}
``` python
node.run('python3 -m pip install --user matplotlib')
```
:::

::: {.cell .markdown}
### Set up Jupyter on server
:::

::: {.cell .markdown}
Install Jupyter:
:::

::: {.cell .code}
``` python
node.run('python3 -m pip install --user  jupyter-core jupyter-client jupyter -U --force-reinstall')
```
:::

::: {.cell .markdown}
### Retrieve the materials

Finally, get a copy of the notebooks that you will run:
:::

::: {.cell .code}
``` python
node.run('git clone https://github.com/indianspeedster/edu-ml.git')
```
:::

::: {.cell .markdown}
## Run a JupyterHub server
:::

::: {.cell .markdown}
Run the following cell:
:::

::: {.cell .code}
``` python
print('ssh -L 127.0.0.1:8888:127.0.0.1:8888 cc@' + reserved_fip) 
```
:::

::: {.cell .markdown}
then paste its output into a *local* terminal on your own device, to set up a tunnel to the Jupyter server. If your Chameleon key is not in the default location, you should also specify the path to your key as an argument, using `-i`. Leave this SSH session open.
:::

::: {.cell .markdown}
Then, run the following cell, which will start a command that does not terminate:
:::

::: {.cell .code}
``` python
node.run("/home/cc/.local/bin/jupyter notebook --port=8888 --notebook-dir='/home/cc/edu-ml/notebooks'")
```
:::

::: {.cell .markdown}
In the output of the cell above, look for a URL in this format:

    http://localhost:8888/?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
:::

::: {.cell .markdown}
Copy this URL and open it in a browser. Then, you can run the sequence of notebooks that you\'ll see there, in order.
:::

::: {.cell .markdown}
If you need to stop and re-start your Jupyter server,

-   Use Kernel \> Interrupt Kernel *twice* to stop the cell above
-   Then run the following cell to kill whatever may be left running in the background.
:::

::: {.cell .code}
``` python
node.run("sudo killall jupyter-notebook")
```
:::

::: {.cell .markdown}
## Release resources

If you finish with your experimentation before your lease expires,release your resources and tear down your environment by running the following (commented out to prevent accidental deletions).

This section is designed to work as a \"standalone\" portion - you can come back to this notebook, ignore the top part, and just run this section to delete your reasources.
:::

::: {.cell .code}
``` python
# setup environment - if you made any changes in the top part, make the same changes here
import chi, os
from chi import lease, server

PROJECT_NAME = os.getenv('OS_PROJECT_NAME')
chi.use_site("CHI@UC")
chi.set("project_name", PROJECT_NAME)

lease = chi.lease.get_lease(f"{os.getenv('USER')}-edu-ml")
```
:::

::: {.cell .code}
``` python
DELETE = False
# DELETE = True 

if DELETE:
    # delete server
    server_id = chi.server.get_server_id(f"{os.getenv('USER')}-deep-nets")
    chi.server.delete_server(server_id)

    # release floating IP
    reserved_fip =  chi.lease.get_reserved_floating_ips(lease["id"])[0]
    ip_info = chi.network.get_floating_ip(reserved_fip)
    chi.neutron().delete_floatingip(ip_info["id"])

    # delete lease
    chi.lease.delete_lease(lease["id"])
```
:::
