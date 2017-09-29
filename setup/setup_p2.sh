#!/bin/bash
#
# Configure a p2.xlarge instance

# get the correct ami
export region=$(aws configure get region)
if [ $region = "us-west-2" ]; then
   export ami="ami-06e80e7c" # Oregon
elif [ $region = "eu-west-1" ]; then
   export ami="ami-06e80e7c" # Ireland
elif [ $region = "us-east-1" ]; then
  export ami="ami-06e80e7c" # Virginia
else
  echo "Only us-west-2 (Oregon), eu-west-1 (Ireland), and us-east-1 (Virginia) are currently supported"
  exit 1
fi

export instanceType="p2.xlarge"

. $(dirname "$0")/setup_instance.sh
