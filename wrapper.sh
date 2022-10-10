# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.


num_efa_devices=`lstopo | grep OpenFabrics | cut -d'"' -f2 | wc -l`
efa_device_idx=1
if [ "$num_efa_devices" -eq "0" ]; then # Mostly will be p3.16 block
  echo "no EFA detected. Skipping RDMA test"
  exit 1
elif [ "$num_efa_devices" -eq "1" ]; then
  efa_device_idx=1
else
  efa_device_idx=$(($OMPI_COMM_WORLD_LOCAL_RANK / 2))
  efa_device_idx=$((efa_device_idx + 1))
fi

export SMDATAPARALLEL_DEVICE_NAME=`lstopo | grep OpenFabrics | cut -d'"' -f2 | sed -n "${efa_device_idx},${efa_device_idx} p"`

$@

