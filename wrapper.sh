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

