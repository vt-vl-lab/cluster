## VLLAB Management

Thanks our lab alumni [Mark](https://markdtw.github.io/) for the technical support!

### How to add a new user
```bash
sudo useradd -m -g 1000 -s /bin/bash <username>
```
**Explanation:** this adds <username> to group ID 1000 (vllab group). This must be done to let <username> to have access to /mnt/data (the HDD)

`sudo chfn <username>		# User information`

`sudo passwd <username>	# Must setup a user password initially`


### FAQ
1) Encounting error "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running."
```bash
sudo service lightdm stop
/home/vllabX/Downloads/> sudo bash NVIDIA-Linux-x86-...run
sudo service lightdm start
```
**Explanation:** whenever the linux kernel is modified/updated (either manually or automatically), the NVIDIA driver wouldn't be able to find the kernel. Just re-install the driver would be sufficient to fix the problem
