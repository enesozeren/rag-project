## Hardware and System Configuration
We apply a limit on the hardware available to each participant to run their solutions.

- All solutions will be run on [AWS g4dn.12xlarge](https://aws.amazon.com/ec2/instance-types/g4/) instances equipped with [NVIDIA T4 GPUs](https://www.nvidia.com/en-us/data-center/tesla-t4/). 
- The hardware available is: 
    - `4` x [NVIDIA T4 GPU](https://www.nvidia.com/en-us/data-center/tesla-t4/s). 
    - `40` x vCPU (`20` physical CPU cores)
    - `180GB` RAM

**Note**: When running in `gpu:false` mode, you will have access to `4` x vCPUs (`2` physical cores) and `8GB` RAM. 

Please note that NVIDIA T4 uses a somewhat outdated architecture and is thus not compatible with certain acceleration toolkits (e.g. Flash Attention), so please be careful about compatibility.

Besides, the following restrictions will also be imposed: 

- Network connection will be disabled.
- Each submission will be assigned a certain amount of time to run. Submissions that exceed the time limits will be killed and will not be evaluated. The tentative time limit is set as follows : 
- `30s` timeout for each sample prediction.
- Overall timeout of `7620s` fot 254 samples.

- Each team will be able to make up to **4 submissions per week per track**, and will be allowed an additional quota of upto **4 failed submissions per task per week**.




